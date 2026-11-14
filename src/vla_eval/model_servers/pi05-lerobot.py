# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vla-eval",
#     "lerobot[pi]>=0.5.0",
#     "numpy>=1.24",
#     "torch",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# # Script-local only: lerobot needs pyarrow>=18; harness pins <18 for calib export.
# # This override does not change the project pyproject.toml / 3.11 env.
# override-dependencies = ["pyarrow>=18"]
# exclude-newer = "2026-11-13T00:00:00Z"
# ///
"""Native LeRobot pi0.5 model server for RoboCasa comparisons.

Loads a LeRobot-format pi0.5 checkpoint (config.json + processors +
model.safetensors) via ``PI05Policy`` / ``make_pre_post_processors``.
Isolated from the harness's Python 3.11 env through this script's PEP 723
metadata (LeRobot >=0.5 needs Python >=3.12).

Aligned with ``configs/model_servers/phyai/pi05/robocasa.yaml`` where it
matters for fair success-rate comparison: same checkpoint, chunk_size=50,
reproducible flow-matching noise (same seed hash scheme as pi05-phyai).
PhyAI-only knobs (QVLA pack, flashinfer, CUDA graph, engine batching) are
intentionally not mirrored.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

NoiseMode = Literal["random", "global", "episode"]

_CAMERA_KEYS = (
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
)


def _parse_noise_mode(value: str) -> NoiseMode:
    mode = value.strip().lower()
    if mode not in ("random", "global", "episode"):
        raise ValueError(
            f"noise_mode must be random|global|episode, got {value!r}."
        )
    return mode  # type: ignore[return-value]


class Pi05LeRobotModelServer(PredictModelServer):
    """pi0.5 server using HuggingFace LeRobot's native policy stack."""

    def __init__(
        self,
        checkpoint_path: str = "/data/share/pi05_robocasa",
        device: str = "cuda",
        dtype: str = "bfloat16",
        reproducible: bool = True,
        inference_seed: int = 10,
        noise_mode: str = "episode",
        compile_model: bool = False,
        num_inference_steps: int | None = None,
        *,
        chunk_size: int = 50,
        action_ensemble: str = "newest",
        max_batch_size: int = 4,
        **kwargs: Any,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}.")
        super().__init__(
            chunk_size=chunk_size,
            action_ensemble=action_ensemble,
            max_batch_size=max_batch_size,
            **kwargs,
        )
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = device
        self.dtype = dtype.strip().lower()
        self.reproducible = bool(reproducible)
        self.noise_mode = _parse_noise_mode(noise_mode)
        self.compile_model = bool(compile_model)
        self.num_inference_steps = num_inference_steps
        if self.noise_mode in ("global", "episode") and not self.reproducible:
            raise ValueError(
                f"noise_mode={self.noise_mode!r} requires reproducible=True."
            )
        if type(inference_seed) is not int:
            raise ValueError(
                f"inference_seed must be an int, got {inference_seed!r}."
            )
        self.inference_seed: int | None = (
            inference_seed if self.reproducible else None
        )
        if self.noise_mode in ("global", "episode") and self.inference_seed is None:
            raise ValueError(
                f"noise_mode={self.noise_mode!r} requires inference_seed."
            )

        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        self._model_chunk_size: int = int(chunk_size)
        self._max_action_dim: int = 32
        self._action_dim: int = 12
        self._state_dim: int = 16
        self._image_keys: tuple[str, ...] = tuple(
            f"observation.images.{name}" for name in _CAMERA_KEYS
        )
        # session_id -> (task_id, episode_idx); only for noise_mode="episode".
        self._episode_keys: dict[str, tuple[int, int]] = {}

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def _load_model(self) -> None:
        if self._policy is not None:
            return

        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Config, PI05Policy

        if not self.checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not (self.checkpoint_path / "config.json").is_file():
            raise FileNotFoundError(
                f"{self.checkpoint_path} is missing config.json "
                "(expected a LeRobot pi0.5 checkpoint)."
            )

        logger.info("Loading LeRobot pi0.5 from %s", self.checkpoint_path)
        cfg = PI05Config.from_pretrained(str(self.checkpoint_path))
        cfg.device = self.device
        cfg.compile_model = self.compile_model
        # Inference-only: checkpoint may enable this for training; it slows init.
        cfg.gradient_checkpointing = False
        # PI05Config.dtype must be a torch.dtype (not the JSON string "bfloat16").
        if self.dtype in ("bfloat16", "bf16"):
            cfg.dtype = torch.bfloat16
        elif self.dtype in ("float32", "fp32"):
            cfg.dtype = torch.float32
        else:
            raise ValueError(
                f"Unsupported dtype={self.dtype!r}; PI05 only accepts bfloat16|float32."
            )
        if self.num_inference_steps is not None:
            cfg.num_inference_steps = int(self.num_inference_steps)

        self._model_chunk_size = int(cfg.chunk_size)
        self._max_action_dim = int(cfg.max_action_dim)
        action_feat = cfg.output_features.get("action")
        if action_feat is not None:
            self._action_dim = int(action_feat.shape[0])
        state_feat = cfg.input_features.get("observation.state")
        if state_feat is not None:
            self._state_dim = int(state_feat.shape[0])
        image_keys = [
            key
            for key, feat in cfg.input_features.items()
            if key.startswith("observation.images.")
        ]
        if image_keys:
            # Prefer RoboCasa training order when present; else config order.
            preferred = [f"observation.images.{name}" for name in _CAMERA_KEYS]
            ordered = [key for key in preferred if key in image_keys]
            ordered.extend(key for key in image_keys if key not in ordered)
            self._image_keys = tuple(ordered)

        # Harness open-loop buffer should match policy n_action_steps.
        n_action_steps = int(getattr(cfg, "n_action_steps", self._model_chunk_size))
        self.chunk_size = n_action_steps

        # Prefer the native safetensors→device path when available (faster than the
        # OpenPI conversion loader that materializes the full state dict on CPU).
        if hasattr(PI05Policy, "use_native_pretrained_loader"):
            PI05Policy.use_native_pretrained_loader = True

        # LeRobot builds nn.Module params with the process default device (CPU unless
        # changed), then .to(cuda). Allocate directly on GPU to skip the CPU copy.
        prev_default_device = None
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            prev_default_device = torch.get_default_device()
            torch.set_default_device(torch.device(self.device))
            logger.info("Default tensor device set to %s for model build/load", self.device)

        logger.info(
            "Building/loading policy on %s (GPU mem should rise immediately)...",
            self.device,
        )
        try:
            self._policy = PI05Policy.from_pretrained(
                str(self.checkpoint_path),
                config=cfg,
                strict=True,
                local_files_only=True,
            )
        finally:
            if prev_default_device is not None:
                torch.set_default_device(prev_default_device)

        logger.info("Policy weights loaded; switching to eval mode")
        self._policy.eval()
        self._policy.to(self.device)

        logger.info("Loading pre/post processors...")
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(self.checkpoint_path),
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )

        if self.reproducible:
            assert self.inference_seed is not None
            self._seed_rng(self.inference_seed)
            self._enable_deterministic_algorithms()

        logger.info(
            "LeRobot pi0.5 ready: chunk=%d n_action_steps=%d action_dim=%d "
            "state_dim=%d images=%s reproducible=%s noise_mode=%s seed=%s",
            self._model_chunk_size,
            n_action_steps,
            self._action_dim,
            self._state_dim,
            list(self._image_keys),
            self.reproducible,
            self.noise_mode,
            self.inference_seed,
        )

    @staticmethod
    def _seed_rng(seed: int) -> None:
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("RNG seeded (seed=%s).", seed)

    @staticmethod
    def _enable_deterministic_algorithms() -> None:
        import torch

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Deterministic algorithms enabled (warn_only=True).")

    def _sample_noise_from_seed(self, seed: int):
        import torch

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) % (2**31))
        return torch.randn(
            1,
            self._model_chunk_size,
            self._max_action_dim,
            generator=gen,
            dtype=torch.float32,
        )

    def _sample_noise_from_unseed(self, batch_size: int = 1):
        import torch

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int.from_bytes(os.urandom(8), "little"))
        return torch.randn(
            batch_size,
            self._model_chunk_size,
            self._max_action_dim,
            generator=gen,
            dtype=torch.float32,
        )

    @staticmethod
    def _episode_noise_seed(
        base_seed: int, task_id: int, episode_idx: int, step: int
    ) -> int:
        """Same hash scheme as ``pi05-phyai.PhyAIModelServer._episode_noise_seed``."""
        payload = f"{int(base_seed)}:{int(task_id)}:{int(episode_idx)}:{int(step)}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % (2**31)

    def _resolve_noise(self, ctx_batch: list[SessionContext]):
        """Build FM noise ``(B, chunk, max_action_dim)`` for a batch of sessions."""
        n = len(ctx_batch)
        if self.noise_mode == "random":
            return self._sample_noise_from_unseed(n)
        assert self.inference_seed is not None
        if self.noise_mode == "global":
            return (
                self._sample_noise_from_seed(int(self.inference_seed))
                .expand(n, -1, -1)
                .contiguous()
            )
        noises = []
        for ctx in ctx_batch:
            key = self._episode_keys.get(ctx.session_id)
            if key is None:
                raise ValueError(
                    "noise_mode='episode' requires EPISODE_START before predict "
                    f"(missing task_id/episode_idx for session={ctx.session_id[:8]})."
                )
            task_id, episode_idx = key
            seed = self._episode_noise_seed(
                int(self.inference_seed), task_id, episode_idx, ctx.step
            )
            noises.append(self._sample_noise_from_seed(seed))
        import torch

        return torch.cat(noises, dim=0)

    async def on_episode_start(
        self, config: dict[str, Any], ctx: SessionContext
    ) -> None:
        await super().on_episode_start(config, ctx)
        # Do not reset shared policy/processors here: concurrent sessions share
        # one GPU model. Chunk buffers are per-session (handled by the base class).
        if self.noise_mode != "episode":
            return
        task = config.get("task")
        if not isinstance(task, dict):
            raise ValueError(
                "noise_mode='episode' requires EPISODE_START payload "
                f"{{'task': {{...}}}}, got config keys={sorted(config.keys())}."
            )
        if "task_id" not in task or "episode_idx" not in task:
            raise ValueError(
                "noise_mode='episode' requires task_id and episode_idx in "
                f"EPISODE_START task dict; got keys={sorted(task.keys())}."
            )
        self._episode_keys[ctx.session_id] = (
            int(task["task_id"]),
            int(task["episode_idx"]),
        )

    async def on_episode_end(
        self, result: dict[str, Any], ctx: SessionContext
    ) -> None:
        self._episode_keys.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    def _fit_state(self, raw_state: Any) -> np.ndarray:
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        dim = self._state_dim
        if state.size == dim:
            return state
        if state.size > dim:
            return np.ascontiguousarray(state[:dim])
        fitted = np.zeros(dim, dtype=np.float32)
        fitted[: state.size] = state
        return fitted

    @staticmethod
    def _hwc_uint8_to_chw_float(image: np.ndarray):
        import torch

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
        return tensor.permute(2, 0, 1).to(torch.float32) / 255.0

    def _observation_to_sample(self, obs: Observation) -> dict[str, Any]:
        """Build an unbatched LeRobot sample (CHW float images, 1-D state)."""
        import torch

        images_dict = obs.get("images", {})
        if not isinstance(images_dict, dict):
            images_dict = {}

        sample: dict[str, Any] = {}
        blank: np.ndarray | None = None
        for key in self._image_keys:
            cam = key.split("observation.images.", 1)[-1]
            frame = images_dict.get(cam)
            if frame is None:
                # Fall back to insertion order if named keys are absent.
                if images_dict and blank is None:
                    first = np.asarray(next(iter(images_dict.values())))
                    blank = np.zeros_like(first)
                if blank is None:
                    blank = np.zeros((256, 256, 3), dtype=np.uint8)
                frame = blank
            sample[key] = self._hwc_uint8_to_chw_float(frame)

        raw_state = obs.get("states", obs.get("state", obs.get("joint_state")))
        if raw_state is None:
            raw_state = np.zeros(self._state_dim, dtype=np.float32)
        sample["observation.state"] = torch.from_numpy(self._fit_state(raw_state))
        sample["task"] = str(obs.get("task_description", "") or "")
        return sample

    def _stack_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack unbatched samples into a true batch for the preprocessor."""
        import torch

        batch: dict[str, Any] = {
            "task": [sample["task"] for sample in samples],
            "observation.state": torch.stack(
                [sample["observation.state"] for sample in samples], dim=0
            ),
        }
        for key in self._image_keys:
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
        return batch

    def _postprocess_chunk(self, actions):
        """Unnormalize ``(T, D)`` or ``(B, T, D)`` via the LeRobot postprocessor."""
        import torch

        assert self._postprocessor is not None
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        # MEAN_STD unnormalize broadcasts over leading dims.
        out = self._postprocessor(actions)
        if isinstance(out, torch.Tensor):
            return out.detach().to(torch.float32).cpu().numpy()
        return np.asarray(out, dtype=np.float32)

    def predict_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> list[Action]:
        import torch

        self._load_model()
        assert self._policy is not None
        assert self._preprocessor is not None

        n = len(obs_batch)
        if len(ctx_batch) != n:
            raise ValueError(
                f"obs_batch/ctx_batch length mismatch: {n} vs {len(ctx_batch)}"
            )
        if n == 0:
            return []

        samples = [self._observation_to_sample(obs) for obs in obs_batch]
        processed = self._preprocessor(self._stack_samples(samples))
        noise = self._resolve_noise(ctx_batch).to(
            device=next(self._policy.parameters()).device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            # (B, T, action_dim) — action dim already trimmed inside policy.
            raw = self._policy.predict_action_chunk(processed, noise=noise)
        chunks = raw[:, : self.chunk_size, : self._action_dim]
        actions = self._postprocess_chunk(chunks)
        if actions.ndim == 2:
            # Postprocessor may collapse a singleton batch; restore (B, T, D).
            actions = actions[None, ...]
        if actions.ndim != 3 or actions.shape[0] != n:
            raise RuntimeError(
                f"Unexpected postprocessed action shape {actions.shape} for batch={n}"
            )
        logger.info("lerobot batch actions shape: %s (B=%d)", actions.shape, n)
        return [{"actions": actions[i]} for i in range(n)]

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return self.predict_batch([obs], [ctx])[0]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(Pi05LeRobotModelServer)
