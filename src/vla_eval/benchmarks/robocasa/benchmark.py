"""RoboCasa benchmark implementation.

RoboCasa is a large-scale simulation framework for kitchen manipulation
tasks built on top of robosuite v2 and MuJoCo.  It provides 365 tasks
(atomic + composite) across 2500+ procedurally-generated kitchen scenes.

Actions are 7-D by default when using the ``PandaOmron`` robot with a
standard ``OSC_POSE`` composite controller: ``[dx, dy, dz, drx, dry, drz,
gripper]``.

Observations expose RGB images from configurable cameras (default:
``robot0_agentview_left`` and ``robot0_eye_in_hand``) plus a natural
language task description obtained via ``env.get_ep_meta()["lang"]``.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("MUJOCO_GL", "egl")

logger = logging.getLogger(__name__)

# This checkpoint's 16-D state matches relative EE, then base pose, then gripper.
# Action is already the env order: arm 6, gripper, base 3, torso, base mode.
_LEROBOT_STATE_KEYS = (
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_gripper_qpos",
)


def lerobot12_to_env_action(flat: np.ndarray, action_dim: int) -> np.ndarray:
    """Pass a 12-D action through in controller order: arm, gripper, base, torso, mode."""
    action = np.asarray(flat, dtype=np.float64).reshape(-1)
    if action.shape[0] != 12:
        raise ValueError(f"lerobot12 action must be 12-D, got {action.shape[0]}.")
    if action_dim < 12:
        return action[:action_dim]
    if action_dim == 12:
        return action
    out = np.zeros(action_dim, dtype=np.float64)
    out[:12] = action
    return out


# This image's object sampler accepts only "A" / "B". Newer configs say pretrain / target.
_LEGACY_OBJ_SPLITS = {"pretrain": "A", "target": "B"}


def _call_create_env(create_env: Any, **kwargs: Any) -> Any:
    """Drop kwargs this robocasa build does not accept.

    Older images use ``obj_instance_split`` (``A`` / ``B``) instead of ``split``.
    """
    accepted = set(inspect.signature(create_env).parameters)
    if "split" in kwargs and "split" not in accepted and "obj_instance_split" in accepted:
        value = kwargs.pop("split")
        kwargs["obj_instance_split"] = _LEGACY_OBJ_SPLITS.get(value, value)
    dropped = [key for key in kwargs if key not in accepted]
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        logger.warning("create_env does not accept %s; omitting those arguments.", ", ".join(dropped))
    return create_env(**kwargs)


def lerobot16_state(raw_obs: dict[str, Any]) -> np.ndarray:
    """16-D proprio: base pose, relative EE pose, gripper qpos."""
    missing = [key for key in _LEROBOT_STATE_KEYS if key not in raw_obs]
    if missing:
        raise KeyError(
            "RoboCasa observation is missing proprio keys "
            f"{missing}. Available keys: {sorted(raw_obs)}."
        )
    return np.concatenate(
        [np.asarray(raw_obs[key], dtype=np.float32).reshape(-1) for key in _LEROBOT_STATE_KEYS]
    )

# Subset of atomic tasks suitable for quick evaluation.
DEFAULT_TASKS = [
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToSink",
    "OpenSingleDoor",
    "CloseDoubleDoor",
    "TurnOnSinkFaucet",
    "PreheatOven",
]


class RoboCasaBenchmark(StepBenchmark):
    """RoboCasa kitchen manipulation benchmark.

    Args:
        tasks: List of RoboCasa environment names to evaluate.
            Defaults to a small atomic-task subset.
        robot: Robot model name (default ``"PandaOmron"``).
        camera_names: Camera names for observations.
        camera_size: Camera resolution (square, default 256).
        max_steps: Maximum steps per episode (default 500).
        split: Dataset split — ``"pretrain"`` or ``"target"``.
        seed: Random seed for environment creation.
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "PandaOmron",
        camera_names: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int = 500,
        split: str = "pretrain",
        seed: int | None = None,
        action_dim: int = 7,
        flip_images: bool = True,
    ) -> None:
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._robot = robot
        self._camera_names = camera_names or [
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ]
        self._camera_size = camera_size
        self._max_steps = max_steps
        if action_dim not in (7, 12):
            raise ValueError(f"action_dim must be 7 or 12, got {action_dim!r}.")
        self._split = split
        self._seed = seed
        self._action_dim = action_dim
        self._flip_images = bool(flip_images)
        if action_dim == 12 and camera_names is None:
            self._camera_names = [
                "robot0_agentview_left",
                "robot0_eye_in_hand",
                "robot0_agentview_right",
            ]
        self._env: Any = None
        self._current_task: str | None = None
        self._lang: str = ""

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def get_tasks(self) -> list[Task]:
        return [{"name": task_name, "task_id": index} for index, task_name in enumerate(self._task_names)]

    def _episode_seed(self, task: Task) -> int:
        """Stable seed for one (task, episode) pair.

        Robosuite advances ``env.rng`` on each hard reset. A bare
        ``env.reset()`` therefore makes episode *k* depend on how many
        times this process already reset the same env — which breaks
        under sharding. Reseeding from ``task_id`` + ``episode_idx``
        makes the layout identical for a given pair.
        """
        base = 0 if self._seed is None else int(self._seed)
        task_id = int(task.get("task_id", 0))
        episode_idx = int(task.get("episode_idx", 0))
        return base + 100_000 * task_id + episode_idx

    def reset(self, task: Task) -> Any:
        from robocasa.utils.env_utils import create_env

        task_name = task["name"]
        episode_seed = self._episode_seed(task)

        # Reuse env for same task across episodes
        if self._env is None or self._current_task != task_name:
            if self._env is not None:
                self._env.close()
            self._env = _call_create_env(
                create_env,
                env_name=task_name,
                robots=self._robot,
                camera_names=self._camera_names,
                camera_widths=self._camera_size,
                camera_heights=self._camera_size,
                render_onscreen=False,
                split=self._split,
                seed=episode_seed,
            )
            self._current_task = task_name
        else:
            # Same env object: pin the RNG before the hard reset so this
            # episode_idx does not depend on prior resets in the shard.
            self._env.seed = episode_seed
            self._env.rng = np.random.default_rng(episode_seed)

        self._env.hard_reset = True
        obs = self._env.reset()
        self._lang = self._env.get_ep_meta().get("lang", task_name)
        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raw_action = np.zeros(self._action_dim)
        raw_action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if self._action_dim == 12:
            raw_action = lerobot12_to_env_action(
                raw_action,
                int(self._env.action_spec[0].shape[0]),
            )
        else:
            assert raw_action.shape[-1] == 7, (
                f"Action dimension mismatch: got {raw_action.shape[-1]}, expected 7. "
                "PandaOmron pi0.5 uses action_dim: 12."
            )

        # Pad or truncate to match env action dimension
        act_dim = self._env.action_spec[0].shape[0]
        if raw_action.shape[0] < act_dim:
            raw_action = np.concatenate([raw_action, np.zeros(act_dim - raw_action.shape[0])])
        elif raw_action.shape[0] > act_dim:
            raw_action = raw_action[:act_dim]

        obs, reward, done, info = self._env.step(raw_action)
        success = bool(self._env._check_success())
        info["success"] = success
        return StepResult(obs=obs, reward=float(success), done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        images: dict[str, Any] = {}
        for cam in self._camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                # RoboCasa images are upside-down — flip vertically
                frame = raw_obs[key]
                if self._flip_images:
                    # MuJoCo's default opengl convention stores the image upside down.
                    frame = frame[::-1]
                images[cam] = np.ascontiguousarray(frame)
        obs: Observation = {
            "images": images,
            "task_description": self._lang,
        }
        if self._action_dim == 12:
            obs["states"] = lerobot16_state(raw_obs)
        return obs

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done or step_result.info.get("success", False)

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self._max_steps}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "robot0_agentview_left": IMAGE_RGB,
            "language": LANGUAGE,
        }

    def render(self) -> np.ndarray | None:
        try:
            return self._env.render()
        except Exception:
            return None
