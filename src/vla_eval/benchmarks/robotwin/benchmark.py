"""RoboTwin 2.0 benchmark — dual-arm manipulation on SAPIEN/CuRobo.

Ported from the existing ``vla_evaluation_harness`` implementation
shipped in the ``robotwin`` Docker image.

Non-obvious behaviors:
    - **Expert check**: ``get_tasks()`` optionally runs the oracle planner
      per seed to verify solvability (``skip_expert_check=False``).
    - **Lazy init**: Heavy imports happen on first use, not at construction.
    - **14D action**: dual-arm qpos; 16D inputs are trimmed to 14D.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
import types
from contextlib import contextmanager
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

ROBOTWIN_ROOT = "/app/RoboTwin"
# Each episode owns 100 consecutive seeds. Attempt 0 is the lane start.
# UnStableError tries the next seed in that lane, so the same episode index
# always settles on the same stable scene.
ROBOTWIN_SEED_LANE = 100


def robotwin_episode_seed(task: dict[str, Any], attempt: int) -> int:
    """Seed for one episode attempt. Lanes do not overlap."""
    if attempt < 0 or attempt >= ROBOTWIN_SEED_LANE:
        raise ValueError(f"Seed attempt must be in [0, {ROBOTWIN_SEED_LANE}).")
    episode_idx = int(task.get("episode_idx", 0))
    if "seed_base" in task:
        return int(task["seed_base"]) + episode_idx * ROBOTWIN_SEED_LANE + attempt
    return int(task["seed"]) + attempt

# Motion helpers that must not run during instruction-placeholder recovery.
_MOTION_STUB_NAMES = (
    "move",
    "grasp_actor",
    "place_actor",
    "back_to_origin",
    "move_by_displacement",
    "open_gripper",
    "close_gripper",
    "delay",
    "get_grasp_pose",
    "pick_and_place_block",
)


class _EvalGripperPlanner:
    """Minimal planner shim for eval-only RoboTwin startup.

    RoboTwin's qpos evaluation path still calls ``plan_grippers()`` during
    env setup, but it never uses CuRobo path planning afterwards.  This shim
    keeps gripper interpolation working while avoiding the expensive CuRobo
    warmup in ``Robot.set_planner()``.
    """

    def plan_grippers(self, now_val: float, target_val: float) -> dict[str, Any]:
        num_step = 200
        per_step = (target_val - now_val) / num_step
        vals = np.linspace(now_val, target_val, num_step)
        return {"num_step": num_step, "per_step": per_step, "result": vals}

    def update_point_cloud(self, world_pcd: Any, resolution: float = 0.02) -> None:
        return None

    def plan_path(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo path planning during episode execution.")

    def plan_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo batch planning during episode execution.")


class _LazyOpen3D(types.ModuleType):
    """Import open3d only when one of its attributes is first accessed."""

    def __init__(self) -> None:
        super().__init__("open3d")
        self._real_module: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        if self._real_module is not None:
            return self._real_module

        if sys.modules.get("open3d") is self:
            sys.modules.pop("open3d", None)
        try:
            module = importlib.import_module("open3d")
        except Exception:
            sys.modules["open3d"] = self
            raise
        self.__dict__.update(module.__dict__)
        self._real_module = module
        sys.modules["open3d"] = module
        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


@contextmanager
def _defer_open3d_import(enabled: bool):
    """Defer open3d import during RoboTwin module import when pointclouds are unused."""
    if not enabled:
        yield
        return

    previous = sys.modules.get("open3d")
    proxy = _LazyOpen3D()
    sys.modules["open3d"] = proxy
    try:
        yield
    finally:
        if sys.modules.get("open3d") is proxy:
            if previous is None:
                sys.modules.pop("open3d", None)
            else:
                sys.modules["open3d"] = previous


def _make_fast_set_planner(robot_mod: Any):
    def _set_planner_fast(self: Any, scene: Any = None) -> None:
        self.communication_flag = False
        self.left_planner = _EvalGripperPlanner()
        self.right_planner = _EvalGripperPlanner()

        if self.need_topp:
            self.left_mplib_planner = robot_mod.MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = robot_mod.MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    return _set_planner_fast


@contextmanager
def _patched_robot_set_planner(enabled: bool):
    """Temporarily skip CuRobo planner warmup during env setup."""
    if not enabled:
        yield
        return

    import envs.robot.robot as robot_mod

    original = robot_mod.Robot.set_planner
    robot_mod.Robot.set_planner = _make_fast_set_planner(robot_mod)
    try:
        yield
    finally:
        robot_mod.Robot.set_planner = original


@contextmanager
def _patched_render_setup(enabled: bool):
    """Temporarily use SAPIEN's default shader during env setup."""
    if not enabled:
        yield
        return

    import sapien.render as sapien_render

    originals = {
        "set_camera_shader_dir": sapien_render.set_camera_shader_dir,
        "set_ray_tracing_samples_per_pixel": sapien_render.set_ray_tracing_samples_per_pixel,
        "set_ray_tracing_path_depth": sapien_render.set_ray_tracing_path_depth,
        "set_ray_tracing_denoiser": sapien_render.set_ray_tracing_denoiser,
    }

    def _set_camera_shader_dir_fast(shader_dir: str) -> None:
        originals["set_camera_shader_dir"]("default")

    sapien_render.set_camera_shader_dir = _set_camera_shader_dir_fast
    sapien_render.set_ray_tracing_samples_per_pixel = lambda spp: None
    sapien_render.set_ray_tracing_path_depth = lambda depth: None
    sapien_render.set_ray_tracing_denoiser = lambda name: None
    try:
        yield
    finally:
        for name, func in originals.items():
            setattr(sapien_render, name, func)


def _strip_instruction_markup(text: str) -> str:
    """Remove RoboTwin ``<Place>``-style annotation tags from descriptions."""
    cleaned = re.sub(r"<[^>\n]+>", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


@contextmanager
def _reused_sapien_renderer(holder: dict[str, Any]):
    """Return one ``SapienRenderer`` for every RoboTwin ``setup_demo`` in this process.

    RoboTwin calls ``sapien.core.SapienRenderer()`` inside ``setup_scene``.
    ``close`` does not destroy that object, so a new one per episode exhausts
    Vulkan host memory. Later calls return the renderer stored in ``holder``.
    Ray-tracing settings stay on the shader path RoboTwin already sets.
    """
    import sapien
    import sapien.core as sapien_core

    modules = [module for module in (sapien_core, sapien) if hasattr(module, "SapienRenderer")]
    original = {module: module.SapienRenderer for module in modules}
    factory = _cached_renderer_factory(holder, original[modules[0]])
    for module in modules:
        module.SapienRenderer = factory
    try:
        yield
    finally:
        for module, renderer_cls in original.items():
            module.SapienRenderer = renderer_cls


def _cached_renderer_factory(holder: dict[str, Any], original: Any) -> Any:
    def factory(*args: Any, **kwargs: Any) -> Any:
        renderer = holder.get("renderer")
        if renderer is None:
            renderer = original(*args, **kwargs)
            holder["renderer"] = renderer
        return renderer

    return factory


def _clear_robotwin_scene(env: Any) -> None:
    """Drop the previous scene so a reused renderer does not keep its cameras."""
    if env is None:
        return
    scene = getattr(env, "scene", None)
    if scene is None:
        return
    try:
        scene.clear()
    except Exception:
        pass
    try:
        env.scene = None
    except Exception:
        pass


def _release_robotwin_env(
    env: Any,
    *,
    renderer_holder: dict[str, Any] | None = None,
    drop_renderer: bool = False,
) -> None:
    """Free this episode's scene and keep the process renderer.

    ``drop_renderer`` is only for a Vulkan crash retry and for process
    cleanup. A normal episode keeps the renderer in ``renderer_holder``.
    """
    if env is None:
        if drop_renderer and renderer_holder is not None:
            renderer_holder["renderer"] = None
        return
    _clear_robotwin_scene(env)
    try:
        env.close_env(clear_cache=False)
    except Exception:
        pass
    if renderer_holder is not None and not drop_renderer:
        renderer = getattr(env, "renderer", None)
        if renderer is not None and renderer_holder.get("renderer") is None:
            renderer_holder["renderer"] = renderer
    for name in ("viewer", "cameras", "robot", "scene", "renderer", "engine"):
        if hasattr(env, name):
            try:
                setattr(env, name, None)
            except Exception:
                pass
    if drop_renderer and renderer_holder is not None:
        renderer_holder["renderer"] = None
    try:
        from sapien.render import clear_cache

        clear_cache()
    except Exception:
        pass
    import gc

    gc.collect()


def _is_renderer_crash(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(
        token in text
        for token in (
            "OutOfHostMemory",
            "OutOfDeviceMemory",
            "physical device",
            "SapienRenderer",
            "createFenceUnique",
            "createSemaphoreUnique",
            "cannot create buffer",
        )
    )


def _placeholders_from_play_once(env: Any) -> dict[str, str]:
    """Rebuild ``play_once``'s ``info`` dict without running the motion planner.

    Official eval gets placeholders from a real expert ``play_once``. With
    ``skip_expert_check`` we still need those ids/arm tags for seen templates.
    Stub every motion helper, then call ``play_once`` so assignments like
    ``pick_and_place_block`` (which choose an arm from object pose) still run.
    """
    if env is None:
        return {}

    originals: dict[str, Any] = {}
    try:
        from envs.utils.action import ArmTag

        def _arm_for(actor: Any) -> Any:
            try:
                x = float(np.asarray(actor.get_pose().p).reshape(-1)[0])
            except Exception:
                x = 0.0
            return ArmTag("left" if x < 0 else "right")

        def _pick_and_place_block(block: Any, target_pose: Any = None) -> str:
            return str(_arm_for(block))

        stubs: dict[str, Any] = {
            "move": lambda *args, **kwargs: None,
            "grasp_actor": lambda *args, **kwargs: None,
            "place_actor": lambda *args, **kwargs: None,
            "back_to_origin": lambda *args, **kwargs: None,
            "move_by_displacement": lambda *args, **kwargs: None,
            "open_gripper": lambda *args, **kwargs: None,
            "close_gripper": lambda *args, **kwargs: None,
            "delay": lambda *args, **kwargs: None,
            "get_grasp_pose": lambda *args, **kwargs: None,
            "pick_and_place_block": _pick_and_place_block,
        }
        for name in _MOTION_STUB_NAMES:
            if hasattr(env, name):
                originals[name] = getattr(env, name)
                setattr(env, name, stubs[name])

        if isinstance(getattr(env, "info", None), dict):
            env.info.pop("info", None)

        result = env.play_once()
        raw: dict[str, Any] = {}
        if isinstance(result, dict):
            maybe = result.get("info")
            if isinstance(maybe, dict):
                raw = maybe
        if not raw:
            raw = getattr(env, "info", {}).get("info") or {}
    except Exception as exc:
        logger.warning(
            "Could not recover %s instruction placeholders: %s",
            type(env).__name__,
            exc,
        )
        return {}
    finally:
        for name, func in originals.items():
            setattr(env, name, func)

    return {str(key): str(value) for key, value in raw.items()}


class RoboTwinBenchmark(StepBenchmark):
    """RoboTwin dual-arm manipulation benchmark (SAPIEN/CuRobo).

    Args:
        task_name: RoboTwin task (e.g. ``"grab_roller"``).
        task_config: Config name under ``task_config/`` (default ``"demo_clean"``).
        seed: Base seed index.  Starting seed = ``100000 * (1 + seed)``.
        instruction_type: Instruction variant (``"seen"`` or ``"unseen"``).
        test_num: Number of valid episodes to evaluate.
        skip_expert_check: If ``True``, skip oracle planner verification in
            ``get_tasks()`` (useful for quick smoke tests).
        fast_init: If ``True``, skip CuRobo planner warmup for qpos evaluation
            episodes after task discovery. This preserves the eval path used by
            the harness while substantially reducing cold-start time.
        fast_render: If ``True``, use SAPIEN's default camera shader instead of
            RoboTwin's ray-traced renderer. Faster, but observation fidelity may
            differ from the reference benchmark.
    """

    def __init__(
        self,
        task_name: str | None = None,
        task_config: str = "demo_clean",
        seed: int = 0,
        instruction_type: str = "seen",
        test_num: int = 100,
        skip_expert_check: bool = False,
        fast_init: bool = True,
        fast_render: bool = False,
        task_names: list[str] | None = None,
    ) -> None:
        import re

        super().__init__()
        names = [str(name) for name in (task_names or [])]
        if task_name and task_name not in names:
            names.insert(0, task_name)
        if not names:
            raise ValueError("RoboTwinBenchmark requires task_name or task_names.")
        for name in names:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid task_name: {name!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_config):
            raise ValueError(f"Invalid task_config: {task_config!r}")
        self.task_names = names
        self.task_name = names[0]
        self.task_config = task_config
        self.seed = seed
        self.instruction_type = instruction_type
        self.test_num = test_num
        self.skip_expert_check = skip_expert_check
        self.fast_init = fast_init
        self.fast_render = fast_render
        self._env: Any = None
        self._env_class: Any = None
        self._args: dict[str, Any] | None = None
        self._sapien_renderer: dict[str, Any] = {"renderer": None}

    # -----------------------------------------------------------------
    # Lazy init
    # -----------------------------------------------------------------

    def _init_robotwin(self) -> None:
        """Add RoboTwin paths, load YAML configs, resolve embodiment."""
        if self._args is not None:
            return

        for p in [ROBOTWIN_ROOT, f"{ROBOTWIN_ROOT}/policy", f"{ROBOTWIN_ROOT}/description/utils"]:
            if p not in sys.path:
                sys.path.insert(0, p)

        os.chdir(ROBOTWIN_ROOT)
        import yaml

        config_path = os.path.join(
            ROBOTWIN_ROOT,
            "task_config",
            f"{self.task_config}.yml",
        )
        with open(config_path) as f:
            args: dict[str, Any] = yaml.safe_load(f)

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config

        from envs import CONFIGS_PATH

        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
            _embodiment_types = yaml.safe_load(f)

        def _get_file(etype: str) -> str:
            return _embodiment_types[etype]["file_path"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False

        def _get_config(robot_file: str) -> dict:
            with open(os.path.join(robot_file, "config.yml")) as f:
                return yaml.safe_load(f)

        args["left_embodiment_config"] = _get_config(args["left_robot_file"])
        args["right_embodiment_config"] = _get_config(args["right_robot_file"])

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
            _camera_config = yaml.safe_load(f)

        hcam = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[hcam]["h"]
        args["head_camera_w"] = _camera_config[hcam]["w"]
        args["eval_mode"] = True

        self._args = args
        with _defer_open3d_import(enabled=not args.get("data_type", {}).get("pointcloud", False)):
            envs_module = importlib.import_module(f"envs.{self.task_name}")
        self._env_class = getattr(envs_module, self.task_name)
        logger.info("RoboTwin initialised: tasks=%s", ",".join(self.task_names))

    def _create_env(self) -> Any:
        assert self._env_class is not None
        return self._env_class()

    def cleanup(self) -> None:
        env = self._env
        self._env = None
        _release_robotwin_env(
            env,
            renderer_holder=self._sapien_renderer,
            drop_renderer=True,
        )

    # -----------------------------------------------------------------
    # StepBenchmark interface
    # -----------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        self._init_robotwin()
        assert self._args is not None
        st_seed = 100000 * (1 + self.seed)

        if self.skip_expert_check:
            # One task per name. Episode count comes from episodes_per_task;
            # reset() turns episode_idx into a distinct seed.
            return [
                {
                    "name": name,
                    "suite": "robotwin",
                    "seed_base": st_seed,
                    "episode_idx": 0,
                    "task_id": st_seed,
                    # Placeholder; reset() rebuilds a seen-template instruction.
                    "instruction": f"Perform the {name} task.",
                }
                for name in self.task_names
            ]

        # Full expert check — same protocol as RoboTwin eval_policy.py:
        # only keep seeds where the oracle plans and check_success() pass.
        from generate_episode_instructions import generate_episode_descriptions

        tasks: list[Task] = []
        for task_name in self.task_names:
            self._use_task(task_name)
            env = self._create_env()
            now_seed = st_seed
            episode_idx = 0
            logger.info(
                "Running expert checks for %s from seed %d ...",
                task_name,
                st_seed,
            )
            try:
                while episode_idx < self.test_num:
                    try:
                        _clear_robotwin_scene(env)
                        with _reused_sapien_renderer(self._sapien_renderer):
                            env.setup_demo(
                                now_ep_num=episode_idx,
                                seed=now_seed,
                                is_test=True,
                                **self._args,
                            )
                        episode_info = env.play_once()
                        env.close_env()
                        if env.plan_success and env.check_success():
                            results = generate_episode_descriptions(
                                task_name,
                                [episode_info["info"]],
                                self.test_num,
                            )
                            instruction = str(
                                np.random.default_rng(now_seed).choice(
                                    results[0][self.instruction_type],
                                )
                            )
                            tasks.append(
                                {
                                    "name": task_name,
                                    "suite": "robotwin",
                                    "seed": now_seed,
                                    "episode_idx": episode_idx,
                                    "task_id": now_seed,
                                    "instruction": instruction,
                                    "instruction_ready": True,
                                }
                            )
                            episode_idx += 1
                    except Exception as e:
                        logger.warning(
                            "Expert check failed for %s seed %d: %s",
                            task_name,
                            now_seed,
                            e,
                        )
                        try:
                            env.close_env()
                        except Exception:
                            pass
                    now_seed += 1
            finally:
                _release_robotwin_env(env, renderer_holder=self._sapien_renderer)
        return tasks

    def _use_task(self, task_name: str) -> None:
        """Load the env class for ``task_name``. Embodiment config stays cached."""
        self._init_robotwin()
        if self._env_class is not None and self.task_name == task_name:
            return
        assert self._args is not None
        self.task_name = task_name
        self._args["task_name"] = task_name
        with _defer_open3d_import(enabled=not self._args.get("data_type", {}).get("pointcloud", False)):
            envs_module = importlib.import_module(f"envs.{task_name}")
        self._env_class = getattr(envs_module, task_name)

    def reset(self, task: Task) -> Any:
        self._use_task(str(task["name"]))
        assert self._args is not None

        if self._env is not None:
            previous = self._env
            self._env = None
            _release_robotwin_env(previous, renderer_holder=self._sapien_renderer)

        # skip_expert_check stores seed_base. Episode i searches a private
        # lane of 100 seeds, in order, and keeps the first stable one.
        # The search order is fixed, so a rerun uses the same scene.
        last_error: Exception | None = None
        seed = robotwin_episode_seed(task, 0)
        render_retries = 0
        drop_renderer = False
        attempt = 0
        raw_obs: Any = None
        while attempt < ROBOTWIN_SEED_LANE:
            seed = robotwin_episode_seed(task, attempt)
            if self._env is not None:
                previous = self._env
                self._env = None
                _release_robotwin_env(
                    previous,
                    renderer_holder=self._sapien_renderer,
                    drop_renderer=drop_renderer,
                )
                drop_renderer = False
            self._env = self._create_env()
            try:
                with (
                    _patched_robot_set_planner(self.fast_init),
                    _patched_render_setup(self.fast_render),
                    _reused_sapien_renderer(self._sapien_renderer),
                ):
                    self._env.setup_demo(
                        now_ep_num=task.get("episode_idx", 0),
                        seed=seed,
                        is_test=True,
                        **self._args,
                    )
                # Camera buffers are allocated on the first get_obs. That can
                # raise "cannot create buffer" after setup_demo already passed.
                task["seed"] = seed
                instruction = self._instruction_for_episode(task)
                task["instruction"] = instruction
                self._env.set_instruction(instruction=instruction)
                raw_obs = self._env.get_obs()
                break
            except Exception as exc:
                last_error = exc
                if type(exc).__name__ == "UnStableError":
                    logger.warning(
                        "Unstable RoboTwin seed %s for %s episode %s; "
                        "trying the next seed in this episode's lane.",
                        seed,
                        task["name"],
                        task.get("episode_idx", 0),
                    )
                    attempt += 1
                    continue
                if _is_renderer_crash(exc) and render_retries < 2:
                    render_retries += 1
                    drop_renderer = True
                    logger.warning(
                        "RoboTwin renderer crashed for %s (%s); "
                        "releasing it and retrying seed %s.",
                        task["name"],
                        exc,
                        seed,
                    )
                    continue
                raise
        else:
            assert last_error is not None
            raise last_error
        return raw_obs

    def _instruction_for_episode(self, task: Task) -> str:
        """Use a training-style sentence instead of the placeholder task name.

        Expert-check ``get_tasks()`` already fills a real seen template; keep it.
        ``skip_expert_check`` only has ``"Perform the <task> task."``, so rebuild
        from ``description/task_instruction`` using stubbed ``play_once`` info.
        """
        existing = str(task.get("instruction", "") or "")
        if task.get("instruction_ready") or (
            existing
            and not existing.startswith("Perform the ")
            and "{" not in existing
            and "<" not in existing
        ):
            logger.info("RoboTwin instruction: %s", existing)
            return existing

        path = os.path.join(
            ROBOTWIN_ROOT, "description", "task_instruction", f"{self.task_name}.json"
        )
        try:
            with open(path) as handle:
                spec = json.load(handle)
        except OSError:
            return existing

        episode = _placeholders_from_play_once(self._env)
        seen = list(spec.get("seen") or [])
        if episode and seen:
            from generate_episode_instructions import filter_instructions, replace_placeholders

            matched = filter_instructions(list(seen), episode)
            pool = matched or seen
            template = pool[int(task.get("episode_idx", 0)) % len(pool)]
            text = replace_placeholders(template, episode)
            if text and "{" not in text:
                logger.info("RoboTwin instruction: %s", text)
                return text

        full = _strip_instruction_markup(str(spec.get("full_description") or ""))
        if full and "{" not in full:
            logger.info("RoboTwin instruction: %s", full)
            return full
        return existing

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        act = np.asarray(raw, dtype=np.float64).flatten()
        if len(act) > 14:
            act = act[:14]
        elif len(act) < 14:
            act = np.pad(act, (0, 14 - len(act)))
        assert act.shape[-1] == 14, f"Action dimension mismatch: got {act.shape[-1]}, expected 14"

        self._env.take_action(act, action_type="qpos")
        raw_obs = self._env.get_obs()
        success = bool(self._env.eval_success)
        done = success or (self._env.take_action_cnt >= self._env.step_lim)
        return StepResult(obs=raw_obs, reward=1.0 if success else 0.0, done=done, info={"success": success})

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        return {
            "images": {
                "head_camera": raw_obs["observation"]["head_camera"]["rgb"],
                "left_camera": raw_obs["observation"]["left_camera"]["rgb"],
                "right_camera": raw_obs["observation"]["right_camera"]["rgb"],
            },
            "task_description": raw_obs.get(
                "language",
                task.get("instruction", ""),
            ),
            "joint_state": np.array(raw_obs["joint_action"]["vector"]),
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False)}

    def _official_step_limit(self) -> int:
        """Highest official eval horizon for the tasks in this run.

        RoboTwin's per-task limits live in ``task_config/_eval_step_limit.yml``
        and run from 400 to 1700. A flat cap of 400 stops the long tasks
        before the env's own ``step_lim`` and records them as failures.
        """
        path = os.path.join(ROBOTWIN_ROOT, "task_config", "_eval_step_limit.yml")
        limits: dict[str, int] = {}
        try:
            with open(path) as handle:
                for line in handle:
                    text = line.strip()
                    if not text or text.startswith("#") or ":" not in text:
                        continue
                    key, value = text.split(":", 1)
                    limits[key.strip()] = int(value.strip())
        except OSError:
            return 1700
        selected = [limits.get(name, 400) for name in self.task_names]
        return max(selected) if selected else 400

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self._official_step_limit(),
            "task_name": self.task_name,
            "action_dim": 14,
            "max_episodes_per_task": self.test_num,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        # 14D dual-arm joint positions
        return {
            "joints": DimSpec("joints", 14, "joint_positions"),
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "head_camera": IMAGE_RGB,
            "left_camera": IMAGE_RGB,
            "right_camera": IMAGE_RGB,
            "state": STATE_JOINT,
            "language": LANGUAGE,
        }
