# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "phyai",
#     "phyai-kernel",
#     "phyai-utils-tools",
#     "phyai-ext",
#     "qvla",
#     "numpy>=1.24",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# phyai = { path = "../../../../phyai/phyai", editable = true }
# phyai-kernel = { path = "../../../../phyai/phyai-kernel", editable = true }
# phyai-utils-tools = { path = "../../../../phyai/phyai-utils-tools", editable = true }
# phyai-ext = { path = "../../../../phyai/phyai-ext", editable = true }
# qvla = { path = "../../../../qvla", editable = true }
# [tool.uv]
# exclude-newer = "2026-08-02T00:00:00Z"
# ///
"""PhyAI pi0.5 model server — full-precision or QVLA W4A4 quantization.

Unified replacement for the former ``pi05-phyai.py`` (FP) and
``pi05-phyai-qvla.py`` (QVLA) servers. All pre/post-processing goes through
:class:`phyai_utils_tools.models.pi05.PI05Processor`; engine wiring, warm-up,
and the vla_eval ModelServer surface are shared.

**Mode selection** — pass ``qvla_pack_path`` to enable quantization:

* **FP (default):** omit ``qvla_pack_path`` (or set it to ``null`` in
  YAML). Engine is built with the configured ``use_cuda_graph`` flag.
* **QVLA:** set ``qvla_pack_path`` to a pack built via
  ``qvla/scripts/build_pi05_pack.py``. The engine is built
  in eager mode first so CUDA graphs do not capture bf16 linears before the
  W4A4 layer swap; when ``use_cuda_graph=True``, graphs are recaptured after
  quantization.
* **Differentiable forward:** set ``use_differentiable_forward=True`` to replace
  phyai's non-differentiable ``scheduler.step`` with
  ``qvla.adapters.pi05.differentiable_forward`` (eager runners + autograd kernel
  patches). Requires ``use_cuda_graph=False``; incompatible flags raise.
* **Eager attention:** set ``use_eager_attention=True`` to replace FlashInfer
  with ``qvla.adapters.pi05.attention_grad.patch_attention_for_inference``
  (batched torch math via ``ragged_paged_attn_batched``). Requires
  ``use_cuda_graph=False`` (implied by ``reproducible=True``); mutually exclusive
  with ``use_differentiable_forward``.
* **Reproducible eval:** set ``reproducible=True`` to enable
  ``pad_length_dummy=True``, ``inference_seed`` (default ``0``),
  ``deterministic_algorithms=True``, and an explicit ``noise_mode``:
  ``"random"`` = fresh noise from OS entropy, independent of
  ``inference_seed``; ``"global"`` = one fixed noise from ``inference_seed``;
  ``"episode"`` = per-sample
  ``hash(seed, task_id, episode_idx, step)``. ``noise_mode="global"`` or
  ``"episode"`` requires ``reproducible=True``.
  ``flashinfer_disable_split_kv`` is configurable inside the
  ``reproducible`` branch. Requires ``use_cuda_graph=False`` only when
  ``flashinfer_disable_split_kv=True`` (flashinfer 0.6.x).

Supports vla-eval batch parallel evaluation: set ``max_batch_size > 1`` in the
server config and run sharded benchmarks (``./scripts/run_sharded.sh``). The
framework coalesces concurrent observations into ``predict_batch()``; phyai's
``PI05WS1Scheduler`` runs a single GPU forward pass per batch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import (
    BackendConfig,
    DeviceConfig,
    EngineConfig,
    RuntimeConfig,
    get_engine_config,
    set_engine_config,
)
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request

import phyai_utils_tools.models.pi05.steps_pi05  # noqa: F401
from phyai_utils_tools.models.pi05.processor_pi05 import (
    PI05Processor,
    PI05ProcessedInputs,
)
from phyai_utils_tools.processing.steps.action_steps import SliceActionStep
from phyai_utils_tools.processing.steps.device_steps import DeviceStep
from phyai_utils_tools.processing.transition import IMAGES, STATE, TASK

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

from qvla import enable_quantization
from qvla.adapters.pi05.attention_grad import (
    patch_attention_for_grad,
    patch_attention_for_inference,
)
from qvla.adapters.pi05.differentiable_forward import (
    differentiable_step,
    force_eager_runners,
    patch_action_heads_for_grad,
    patch_expert_ops_for_grad,
    patch_llm_ops_for_grad,
    reset_differentiable_state,
)
from qvla.core.pack import Pack
from qvla.core.quantize import warmup_fp4_cuda_tensors
from qvla.runtime import list_replaced, reset_step_counters


logger = logging.getLogger(__name__)

# Temporary cross-process validation against pack-build calibration.
# Resolved relative to this file: ../../../.. / calibration_data / noise
GLOBAL_CALIBRATION_NOISE_DIR = (
    Path(__file__).resolve().parents[4] / "calibration_data" / "noise"
)


def _global_calibration_noise_path(*, model: str, seed: int) -> Path:
    model_slug = model.strip().lower()
    if not model_slug or any(
        c != "_" and not c.isalnum() for c in model_slug
    ):
        raise ValueError(
            "model must be a non-empty alphanumeric/underscore slug, "
            f"got {model!r}."
        )
    return (
        GLOBAL_CALIBRATION_NOISE_DIR
        / f"qvla_global_noise_{model_slug}_seed{int(seed)}.pt"
    )


NOISE_MODES = ("random", "global", "episode")


def _parse_noise_mode(noise_mode: str) -> str:
    mode = noise_mode.strip().lower()
    if mode not in NOISE_MODES:
        raise ValueError(
            f"noise_mode must be one of {NOISE_MODES}, got {noise_mode!r}."
        )
    return mode


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _format_step_latency_stats(
    times_ms: list[float],
    *,
    n_timed: int,
    n_warmup: int | None = None,
) -> str:
    mean = statistics.fmean(times_ms)
    median = statistics.median(times_ms)
    stdev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    label = (
        f"(n_warmup={n_warmup}, n_timed_dummy={n_timed})"
        if n_warmup is not None
        else f"(n_timed_real={n_timed})"
    )
    return (
        f"mean={mean:.2f} ms  "
        f"median={median:.2f} ms  "
        f"std={stdev:.2f} ms  "
        f"min={min(times_ms):.2f} ms  "
        f"max={max(times_ms):.2f} ms  "
        f"{label}"
    )


def _benchmark_engine_step(
    step_fn,
    request: PI05Request,
    *,
    n_warmup: int,
    n_timed_dummy: int,
) -> str | None:
    """Warm + ``n_timed_dummy`` step calls; return formatted latency or None."""
    for _ in range(n_warmup):
        step_fn(request)
    torch.cuda.synchronize()

    if n_timed_dummy <= 0:
        return None

    times_ms: list[float] = []
    for _ in range(n_timed_dummy):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn(request)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return _format_step_latency_stats(times_ms, n_warmup=n_warmup, n_timed=n_timed_dummy)


class PhyAIModelServer(PredictModelServer):
    """pi0.5 server — FP by default; QVLA when ``qvla_pack_path`` is set."""

    def __init__(
        self,
        checkpoint_path: str,
        noise_mode: str,
        model_kind: str,
        device: str = "cuda",
        params_dtype: str = "bfloat16",
        vision_params_dtype: str = "float16",
        use_cuda_graph: bool = True,
        attn_backend: str = "flashinfer",
        norm_backend: str = "flashinfer",
        linear_backend: str | None = None,
        max_batch_size: int = 1,
        qvla_pack_path: str | None = None,
        strict_pack_match: bool = True,
        use_eager_attention: bool = False,
        use_differentiable_forward: bool = False,
        reproducible: bool = False,
        inference_seed: int = 0,
        embodiment: str | None = None,
        openpi_norm_stats: str | None = None,
        openpi_chunk_size: int | None = None,
        openpi_action_dim: int | None = None,
        openpi_state_dim: int | None = None,
        openpi_empty_cameras: int | None = None,
        openpi_num_inference_steps: int = 10,
        openpi_n_action_steps: int | None = None,
        openpi_discrete_state: bool = True,
        n_warmup: int = 3,
        n_timed_dummy: int = 3,
        n_timed_real: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        self.embodiment = embodiment.strip().lower() if embodiment else ""
        if self.embodiment not in ("", "robotwin"):
            raise ValueError(
                "embodiment must be robotwin or omitted, "
                f"got {embodiment!r}."
            )
        self.openpi_norm_stats = openpi_norm_stats
        self.openpi_chunk_size = openpi_chunk_size
        self.openpi_action_dim = openpi_action_dim
        self.openpi_state_dim = openpi_state_dim
        self.openpi_empty_cameras = openpi_empty_cameras
        self.openpi_num_inference_steps = openpi_num_inference_steps
        self.openpi_n_action_steps = openpi_n_action_steps
        self.openpi_discrete_state = bool(openpi_discrete_state)
        self.model_kind = model_kind.strip().lower()
        if not self.model_kind or any(
            c != "_" and not c.isalnum() for c in self.model_kind
        ):
            raise ValueError(
                "model_kind must be a non-empty alphanumeric/underscore slug, "
                f"got {model_kind!r}."
            )
        self.device = device
        self.params_dtype = _DTYPE_MAP[params_dtype]
        self.vision_params_dtype = _DTYPE_MAP[vision_params_dtype]
        self.use_cuda_graph = bool(use_cuda_graph)
        # Bundled by ``reproducible`` (toggle individually inside the branch
        # for ablations):
        #   True  -> length-dummy pad, inference_seed, deterministic_algorithms
        #   False -> defaults (no pad, inference_seed ignored)
        # ``noise_mode`` selects flow-matching noise:
        #   random  -> fresh noise from OS entropy, independent of inference_seed
        #   global  -> one fixed noise from inference_seed
        #   episode -> hash(seed, task_id, episode_idx, step) per sample
        # flashinfer_disable_split_kv is set in the reproducible branch below.
        # disable_split_kv + use_cuda_graph is incompatible (checked in _load_model).
        self.reproducible = bool(reproducible)
        self.noise_mode = _parse_noise_mode(noise_mode)
        if self.noise_mode in ("global", "episode") and not self.reproducible:
            raise ValueError(
                f"noise_mode={self.noise_mode!r} requires reproducible=True."
            )
        # YAML ``max_batch_size`` is always the engine size (prefer power of 2).
        # reproducible reserves 1 slot for the length dummy → business bs - 1.
        self._engine_max_batch_size = max_batch_size
        if self.reproducible:
            if max_batch_size < 2:
                raise ValueError(
                    "reproducible=True requires max_batch_size >= 2 "
                    "(one slot is reserved for the length-dummy pad)."
                )
            self.flashinfer_disable_split_kv = False
            self.pad_length_dummy = True
            if type(inference_seed) is not int:
                raise ValueError(
                    "inference_seed must be an int, "
                    f"got {inference_seed!r}."
                )
            self.inference_seed: int | None = inference_seed
            self.deterministic_algorithms = True
            self.max_batch_size = max_batch_size - 1
        else:
            self.flashinfer_disable_split_kv = False
            self.pad_length_dummy = False
            self.inference_seed = None
            self.deterministic_algorithms = False
            self.max_batch_size = max_batch_size
        if self.noise_mode in ("global", "episode") and self.inference_seed is None:
            raise ValueError(
                f"noise_mode={self.noise_mode!r} requires inference_seed "
                "(set reproducible=True)."
            )
        self.attn_backend = attn_backend
        self.norm_backend = norm_backend
        self.linear_backend = linear_backend
        self.qvla_pack_path = (
            Path(qvla_pack_path) if qvla_pack_path else None
        )
        self.strict_pack_match = bool(strict_pack_match)
        self.use_eager_attention = bool(use_eager_attention)
        self.use_differentiable_forward = bool(use_differentiable_forward)
        self.n_warmup = n_warmup
        self.n_timed_dummy = n_timed_dummy
        self.n_timed_real = n_timed_real
        # session_id -> (task_id, episode_idx); only used when noise_mode="episode".
        self._episode_keys: dict[str, tuple[int, int]] = {}

        self._config: dict[str, Any] | None = None
        self._processor: PI05Processor | None = None
        self._engine: Engine | None = None
        self._real_times_ms: list[float] = []

        self._image_size: int = 224
        self._num_real_cameras: int = 2
        self._action_dim: int = 7
        self._state_dim: int = 8
        self._tokenizer_max_length: int = 200
        self._model_chunk_size: int = 50
        self._max_action_dim: int = 32

        logger.info(
            "reproducible=%s use_cuda_graph=%s flashinfer_disable_split_kv=%s "
            "pad_length_dummy=%s inference_seed=%s deterministic_algorithms=%s "
            "noise_mode=%s max_batch_size=%d engine_bs=%d",
            self.reproducible,
            self.use_cuda_graph,
            self.flashinfer_disable_split_kv,
            self.pad_length_dummy,
            self.inference_seed,
            self.deterministic_algorithms,
            self.noise_mode,
            self.max_batch_size,
            self._engine_max_batch_size,
        )

    @property
    def _qvla_enabled(self) -> bool:
        return self.qvla_pack_path is not None

    def _lerobot_pi05_weight_remap(self, key: str) -> str | None:
        """Handle LeRobot checkpoints that wrap keys with an extra model. prefix."""
        if key.startswith("model."):
            key = key[len("model.") :]
        if "embed_tokens" in key or "gemma_expert.lm_head" in key:
            return None
        return key

    def _read_config(self) -> dict[str, Any]:
        if self._config is None:
            with (self.checkpoint_path / "config.json").open("r", encoding="utf-8") as f:
                self._config = json.load(f)
        return self._config

    @staticmethod
    def _reorder_postprocessor_slice_first(processor: PI05Processor) -> None:
        """Put SliceActionStep before UnnormalizerStep in the loaded post chain."""
        steps = list(processor.postprocessor.steps)
        slice_steps = [s for s in steps if isinstance(s, SliceActionStep)]
        core_steps = [s for s in steps if not isinstance(s, SliceActionStep)]
        processor._postprocessor.steps = [*slice_steps, *core_steps]

    def _make_dummy_obs(self) -> dict[str, Any]:
        images = {
            f"image_{index}": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for index in range(self._num_real_cameras)
        }
        return {
            "images": images,
            "states": np.random.randn(self._state_dim).astype(np.float32),
            "task_description": "pad " * self._tokenizer_max_length,
        }

    def _pad_obs_with_length_dummy(
        self, phyai_obs_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Pad to ``_engine_max_batch_size`` with longest-prompt dummies.

        No-op when ``pad_length_dummy`` is False (returns ``phyai_obs_list`` as-is).
        """
        if not self.pad_length_dummy:
            return phyai_obs_list
        n = len(phyai_obs_list)
        if n > self.max_batch_size:
            raise ValueError(
                f"business batch size {n} exceeds max_batch_size={self.max_batch_size}"
            )
        n_pad = self._engine_max_batch_size - n
        return [*phyai_obs_list, *[self._make_dummy_obs() for _ in range(n_pad)]]

    def _obs_to_transition(self, obs: dict[str, Any]) -> dict[str, Any]:
        cameras: list[torch.Tensor] = []
        images = list(obs.get("images", {}).values())
        # OpenPI pi0.5 always has 3 camera token slots. Checkpoints trained with
        # an empty third camera (CALVIN) need that slot present so language
        # tokens stay at the positions the model was trained with. Pad with black.
        if images and len(images) < self._num_real_cameras:
            blank = np.zeros_like(np.asarray(images[0]))
            images = [*images, *([blank] * (self._num_real_cameras - len(images)))]
        for img in images[: self._num_real_cameras]:
            arr = np.ascontiguousarray(img)
            cameras.append(
                torch.as_tensor(arr, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )
        return {
            IMAGES: cameras,
            STATE: torch.as_tensor(
                np.asarray(obs.get("states", np.zeros(self._state_dim)), dtype=np.float32)
            ),
            TASK: str(obs.get("task_description", "")),
        }

    def _build_request(self, obs: dict[str, Any], noise: torch.Tensor | None = None) -> PI05Request:
        return self._build_batch_request([obs], noise=noise)

    def _build_batch_request(
        self,
        obs_list: list[dict[str, Any]],
        noise: torch.Tensor | None = None,
    ) -> PI05Request:
        assert self._processor is not None
        processed: list[PI05ProcessedInputs] = [
            self._processor.preprocess(self._obs_to_transition(obs)) for obs in obs_list
        ]
        return PI05Request(
            pixel_values=torch.cat([p.pixel_values for p in processed], dim=0),
            input_ids=torch.cat([p.input_ids for p in processed], dim=0),
            lang_lens=torch.cat([p.lang_lens for p in processed], dim=0),
            noise=noise,
        )

    def _seed_rng(self) -> None:
        """Seed numpy / torch / CUDA RNGs from ``inference_seed``."""
        seed = int(self.inference_seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("RNG seeded (seed=%s).", seed)

    def _enable_deterministic_algorithms(self) -> None:
        """Force cudnn / cublas / torch deterministic algorithm selection."""
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        logger.info(
            "Deterministic algorithms on. Prefer attn_backend=eager for stable scores."
        )

    def _sample_noise_from_seed(self, seed: int) -> torch.Tensor:
        """Sample one FM noise tensor ``(1, chunk, action_dim)`` from ``seed``."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) % (2**31))
        return torch.randn(
            1,
            self._model_chunk_size,
            self._max_action_dim,
            generator=gen,
            dtype=torch.float32,
        )

    def _sample_noise_from_unseed(self, batch_size: int) -> torch.Tensor:
        """Sample FM noise from OS entropy without consuming global RNG state."""
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
        """Deterministic seed from ``hash(base_seed, task_id, episode_idx, step)``."""
        payload = f"{int(base_seed)}:{int(task_id)}:{int(episode_idx)}:{int(step)}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % (2**31)

    def _make_noise(self, batch_size: int) -> torch.Tensor:
        """Expand one globally seeded flow-matching noise across the batch."""
        if self.noise_mode == "episode":
            raise ValueError(
                "_make_noise() is the global-noise path; with "
                "noise_mode='episode' use _make_episode_bound_noise()."
            )
        if self.inference_seed is None:
            raise ValueError(
                "noise_mode='global' requires inference_seed to be set "
                "(got inference_seed=None)."
            )
        noise = self._sample_noise_from_seed(int(self.inference_seed))
        # Temporary cross-process validation against pack-build calibration.
        if not getattr(self, "_global_noise_validation_done", False):
            path = _global_calibration_noise_path(
                model=self.model_kind, seed=int(self.inference_seed)
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Global calibration noise file not found: {path}"
                )
            calibration_noise = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            if not torch.equal(noise, calibration_noise):
                raise RuntimeError(
                    "Model-server global noise differs from saved calibration "
                    f"noise at {path} "
                    f"(server={tuple(noise.shape)}/{noise.dtype}, "
                    f"calibration={tuple(calibration_noise.shape)}/"
                    f"{calibration_noise.dtype})."
                )
            logger.info(
                "Global noise exactly matches saved calibration noise: %s",
                path,
            )
            self._global_noise_validation_done = True
        return noise.expand(batch_size, -1, -1).contiguous()

    def _make_episode_bound_noise(
        self, ctx_batch: list[SessionContext], n_total: int
    ) -> torch.Tensor:
        """Per-sample noise from ``hash(seed, task_id, episode_idx, step)``.

        Pad slots (``n_total > len(ctx_batch)``) use deterministic pad seeds
        ``hash(seed, -1, pad_i, 0)`` so they never share real-episode seeds.
        """
        if self.inference_seed is None:
            raise ValueError(
                "noise_mode='episode' requires inference_seed to be set "
                "(got inference_seed=None)."
            )
        n_real = len(ctx_batch)
        if n_total < n_real:
            raise ValueError(
                f"n_total={n_total} < len(ctx_batch)={n_real} for episode-bound noise."
            )
        base = int(self.inference_seed)
        noises: list[torch.Tensor] = []
        for ctx in ctx_batch:
            key = self._episode_keys.get(ctx.session_id)
            if key is None:
                raise ValueError(
                    "noise_mode='episode' requires EPISODE_START before "
                    f"predict (missing task_id/episode_idx for session="
                    f"{ctx.session_id[:8]})."
                )
            task_id, episode_idx = key
            seed = self._episode_noise_seed(base, task_id, episode_idx, ctx.step)
            noises.append(self._sample_noise_from_seed(seed))
        for pad_i in range(n_total - n_real):
            seed = self._episode_noise_seed(base, -1, pad_i, 0)
            noises.append(self._sample_noise_from_seed(seed))
        return torch.cat(noises, dim=0)

    def _resolve_noise(
        self, ctx_batch: list[SessionContext], n_total: int
    ) -> torch.Tensor:
        """Select OS-random, global, or episode-bound noise for a batch."""
        if self.noise_mode == "random":
            return self._sample_noise_from_unseed(n_total)
        if self.noise_mode == "episode":
            return self._make_episode_bound_noise(ctx_batch, n_total)
        return self._make_noise(n_total)

    async def on_episode_start(
        self, config: dict[str, Any], ctx: SessionContext
    ) -> None:
        await super().on_episode_start(config, ctx)
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

    def _decode_actions(self, raw_actions: torch.Tensor) -> np.ndarray:
        assert self._processor is not None
        actions_t: torch.Tensor = self._processor.postprocess(raw_actions.detach())
        return actions_t.to(torch.float32).numpy()

    def _step(self, request: PI05Request) -> torch.Tensor:
        """One inference step — ``engine.step`` or qvla ``differentiable_step``."""
        assert self._engine is not None
        if not self.use_differentiable_forward:
            return self._engine.step(request)

        sched = getattr(self._engine.entry, "scheduler", None)
        if sched is None:
            raise RuntimeError(
                "use_differentiable_forward=True but engine has no scheduler."
            )
        reset_step_counters(self._engine.entry.model)
        reset_differentiable_state(sched)
        # Autograd must be on so qvla's kernel patches route to torch backends.
        with torch.enable_grad():
            actions = differentiable_step(sched, request)
        return actions.detach()

    def _engine_step_timed(self, request: PI05Request) -> torch.Tensor:
        assert self._engine is not None
        if self.n_timed_real <= 0 or len(self._real_times_ms) >= self.n_timed_real:
            return self._step(request)

        batch_size = int(request.pixel_values.shape[0])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        raw_actions = self._step(request)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        self._real_times_ms.append(elapsed_ms)
        logger.info(
            "Real-data step latency [%d/%d] B=%d: %.2f ms",
            len(self._real_times_ms),
            self.n_timed_real,
            batch_size,
            elapsed_ms,
        )
        if len(self._real_times_ms) >= self.n_timed_real:
            logger.info(
                "Real-data step latency summary: %s",
                _format_step_latency_stats(self._real_times_ms, n_timed=self.n_timed_real),
            )
        return raw_actions

    def _build_engine(self, *, use_cuda_graph: bool) -> Engine:
        assert self._processor is not None
        return Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=self.checkpoint_path,
                    max_batch_size=self._engine_max_batch_size,
                    vision_params_dtype=self.vision_params_dtype,
                    weight_remap=self._lerobot_pi05_weight_remap,
                    inputs_image_shape=[
                        [self._image_size, self._image_size, 3]
                        for _ in range(self._num_real_cameras)
                    ],
                ),
                config=EngineConfig(
                    backends=BackendConfig(
                        attn=self.attn_backend,
                        norm=self.norm_backend,
                        linear=self.linear_backend,
                    ),
                    device=DeviceConfig(target=self.device, params_dtype=self.params_dtype),
                    runtime=RuntimeConfig(
                        use_cuda_graph=use_cuda_graph,
                        force_linear_kernel=self.linear_backend,
                        flashinfer_workspace_bytes=max(
                            256 * 1024 * 1024,
                            64 * 1024 * 1024 * self._engine_max_batch_size,
                        ),
                        flashinfer_disable_split_kv=self.flashinfer_disable_split_kv,
                    ),
                ),
            )
        )

    def _setup_processor(self) -> None:
        cfg = self._read_config()
        self._image_size = int(cfg.get("image_resolution", [224, 224])[0])
        empty_cameras = int(cfg.get("empty_cameras", 1))
        self._num_real_cameras = max(1, 3 - empty_cameras)
        self._action_dim = int(
            cfg.get("output_features", {}).get("action", {}).get("shape", [7])[0]
        )
        self._state_dim = int(
            cfg.get("input_features", {}).get("observation.state", {}).get("shape", [8])[0]
        )
        self._tokenizer_max_length = int(cfg.get("tokenizer_max_length", 200))
        model_chunk_size = int(cfg.get("chunk_size", 50))
        n_action_steps = int(cfg.get("n_action_steps", model_chunk_size))
        self.chunk_size = n_action_steps
        self._model_chunk_size = model_chunk_size
        self._max_action_dim = int(cfg.get("max_action_dim", 32))
        logger.info(
            "Action chunk buffer: n_action_steps=%d (model chunk_size=%d)",
            n_action_steps,
            model_chunk_size,
        )

        self._tokenizer_name = cfg.get("tokenizer_name", "google/paligemma-3b-pt-224")
        self._processor = PI05Processor.from_pretrained(
            self.checkpoint_path,
            tokenizer_name=self._tokenizer_name,
            image_size=self._image_size,
            num_images=self._num_real_cameras,
            action_dim=self._action_dim,
            normalize_pixels=True,
            device=self.device,
            params_dtype=self.params_dtype,
        )
        self._reorder_postprocessor_slice_first(self._processor)
        for step in self._processor.preprocessor.steps:
            if isinstance(step, DeviceStep):
                step.float_dtype = self.params_dtype
                step._float_dtype = self.params_dtype

    @staticmethod
    def _clear_runner_graphs(runner: Any) -> None:
        graphs = getattr(runner, "graphs", None)
        if isinstance(graphs, dict):
            graphs.clear()
        if getattr(runner, "graph", None) is not None:
            runner.graph = None

    @staticmethod
    def _set_runner_cuda_graph_flag(runner: Any, *, enabled: bool) -> bool:
        if runner is None:
            return False
        device = getattr(runner, "device", None)
        on_cuda = device is not None and torch.device(device).type == "cuda"
        attn_backend = getattr(runner, "attn_backend", None)
        if attn_backend is not None:
            active = bool(enabled) and attn_backend.supports_capture() and on_cuda
        else:
            active = bool(enabled) and on_cuda
        runner.use_cuda_graph = active
        return active

    @classmethod
    def _force_eager_before_quant(cls, engine: Engine) -> None:
        eng = get_engine_config()
        if eng.runtime.use_cuda_graph:
            logger.warning(
                "EngineConfig.use_cuda_graph=True after Engine init (often "
                "PHYAI_USE_CUDA_GRAPH in the environment). Forcing eager mode "
                "before QVLA layer swap."
            )
            set_engine_config(
                eng.replace(runtime=replace(eng.runtime, use_cuda_graph=False))
            )

        sched = getattr(engine.entry, "scheduler", None)
        if sched is None:
            return

        for runner_name in ("vision_runner", "llm_runner", "expert_runner"):
            runner = getattr(sched, runner_name, None)
            if runner is None:
                continue
            if getattr(runner, "use_cuda_graph", False):
                logger.warning("Disabling CUDA graphs on %s before QVLA swap.", runner_name)
                runner.use_cuda_graph = False
            cls._clear_runner_graphs(runner)

    @classmethod
    def _recapture_cuda_graphs_after_quant(cls, engine: Engine) -> None:
        sched = getattr(engine.entry, "scheduler", None)
        if sched is None:
            return

        eng = get_engine_config()
        set_engine_config(
            eng.replace(runtime=replace(eng.runtime, use_cuda_graph=True))
        )

        active_runners: list[str] = []
        for runner_name in ("vision_runner", "llm_runner", "expert_runner"):
            runner = getattr(sched, runner_name, None)
            cls._clear_runner_graphs(runner)
            if cls._set_runner_cuda_graph_flag(runner, enabled=True):
                active_runners.append(runner_name)

        if not active_runners:
            logger.warning(
                "use_cuda_graph=True requested but no runner supports capture; "
                "staying in eager mode."
            )
            set_engine_config(
                eng.replace(runtime=replace(eng.runtime, use_cuda_graph=False))
            )
            return

        logger.info(
            "Recapturing CUDA graphs after QVLA swap on: %s",
            ", ".join(active_runners),
        )
        warmup_fp4_cuda_tensors(get_engine_config().device.target)
        sched.setup()

    def _apply_qvla(self) -> None:
        assert self._engine is not None
        assert self.qvla_pack_path is not None

        logger.info("Loading QVLA pack from %s ...", self.qvla_pack_path)
        pack = Pack.load(self.qvla_pack_path)
        logger.info("Pack header:\n%s", pack.summary())

        replaced = enable_quantization(
            self._engine.entry.model,
            pack,
            strict=self.strict_pack_match,
            nvfp_activation_num_samples=self._engine_max_batch_size,
        )
        n_live = sum(1 for _ in list_replaced(self._engine.entry.model))
        logger.info(
            "QVLA: %d layers swapped in; %d QuantLinear modules live in model.",
            len(replaced),
            n_live,
        )
        if n_live == 0:
            raise RuntimeError(
                "enable_quantization reported success but the model contains zero "
                "QuantLinear modules — inference would stay at full precision."
            )
        if n_live != len(replaced):
            logger.warning(
                "Layer swap count (%d) != live QuantLinear count (%d).",
                len(replaced),
                n_live,
            )

    def _apply_differentiable_forward(self) -> None:
        """Install qvla autograd patches for the lifetime of this server process."""
        assert self._engine is not None
        sched = getattr(self._engine.entry, "scheduler", None)
        if sched is None:
            raise RuntimeError(
                "use_differentiable_forward=True but engine has no scheduler."
            )
        force_eager_runners(sched)
        patch_attention_for_grad(sched, llm=True, expert=True)
        patch_action_heads_for_grad(sched)
        patch_llm_ops_for_grad(sched)
        patch_expert_ops_for_grad(sched)
        logger.info(
            "Differentiable forward enabled: eager runners + qvla autograd patches; "
            "inference uses differentiable_step (not engine.step)."
        )

    def _apply_qvla_attention(self) -> None:
        """Install qvla batched eager attention (no FlashInfer)."""
        assert self._engine is not None
        sched = getattr(self._engine.entry, "scheduler", None)
        if sched is None:
            raise RuntimeError("use_eager_attention=True but engine has no scheduler.")
        patch_attention_for_inference(sched, llm=True, expert=True, batched=True)
        logger.info(
            "Eager attention enabled: qvla patch_attention_for_inference "
            "(batched torch math, no FlashInfer)."
        )

    def _warmup_engine(self) -> None:
        assert self._engine is not None
        dummy_req = self._build_batch_request(
            [self._make_dummy_obs() for _ in range(self._engine_max_batch_size)]
        )
        if self.use_differentiable_forward:
            mode = "differentiable"
        elif self._qvla_enabled:
            mode = "QVLA"
        else:
            mode = "FP"
        logger.info(
            "Preparing request tensors (engine_batch_size=%d, business_max=%d, "
            "pad_length_dummy=%s, mode=%s)...",
            self._engine_max_batch_size,
            self.max_batch_size,
            self.pad_length_dummy,
            mode,
        )
        latency = _benchmark_engine_step(
            self._step,
            dummy_req,
            n_warmup=self.n_warmup,
            n_timed_dummy=self.n_timed_dummy,
        )
        if latency is not None:
            logger.info("%s warm-up complete. step latency: %s", mode, latency)
        else:
            if self._qvla_enabled or self.use_differentiable_forward:
                logger.info("%s warm-up complete.", mode)
            else:
                logger.info("Warm-up complete. The PhyAI brain is now officially online!")

    def _load_model(self) -> None:
        if self._engine is not None:
            return

        from vla_eval.model_servers.pi05_openpi import prepare_pi05_checkpoint

        self.checkpoint_path = prepare_pi05_checkpoint(
            self.checkpoint_path,
            norm_stats=self.openpi_norm_stats,
            chunk_size=self.openpi_chunk_size,
            action_dim=self.openpi_action_dim,
            state_dim=self.openpi_state_dim,
            empty_cameras=self.openpi_empty_cameras,
            num_inference_steps=self.openpi_num_inference_steps,
            n_action_steps=self.openpi_n_action_steps,
            discrete_state_input=self.openpi_discrete_state,
        )

        if self.inference_seed is not None:
            self._seed_rng()
        if self.deterministic_algorithms:
            self._enable_deterministic_algorithms()

        self._setup_processor()

        if self.use_differentiable_forward and self.use_cuda_graph:
            raise ValueError(
                "use_differentiable_forward=True requires use_cuda_graph=False "
                "(CUDA graphs are incompatible with the differentiable forward path)."
            )
        if self.flashinfer_disable_split_kv and self.use_cuda_graph:
            raise ValueError(
                "flashinfer_disable_split_kv=True requires use_cuda_graph=False "
                "(flashinfer 0.6.x: split-kv-off plan path is incompatible with "
                "CUDA graph capture)."
            )
        if self.use_eager_attention and self.use_cuda_graph:
            raise ValueError(
                "use_eager_attention=True requires use_cuda_graph=False "
                "(QVLA attention uses dynamic Python loops and metadata gathers)."
            )
        if self.use_eager_attention and self.use_differentiable_forward:
            raise ValueError(
                "use_eager_attention and use_differentiable_forward are mutually exclusive."
            )

        if self._qvla_enabled:
            self._engine = self._build_engine(use_cuda_graph=False)
            self._force_eager_before_quant(self._engine)
            self._apply_qvla()
            if self.use_cuda_graph:
                self._recapture_cuda_graphs_after_quant(self._engine)
            else:
                sched = getattr(self._engine.entry, "scheduler", None)
                if sched is not None:
                    sched.setup()
            reset_step_counters(self._engine.entry.model)
        else:
            self._engine = self._build_engine(use_cuda_graph=self.use_cuda_graph)

        if self.use_eager_attention:
            self._apply_qvla_attention()

        if self.use_differentiable_forward:
            self._apply_differentiable_forward()

        self._warmup_engine()

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

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

    def _observation_to_phyai_obs(self, obs: Observation) -> dict[str, Any]:
        images_dict = obs.get("images", {})
        if not isinstance(images_dict, dict):
            images_dict = {}

        # Prefer named camera keys so left/right wrists cannot silently swap if
        # make_obs insertion order changes. RoboTwin/Motus train as
        # high → left_wrist → right_wrist (= image_0/1/2 with empty_cameras=0).
        if self.embodiment == "robotwin":
            preferred = ("head_camera", "left_camera", "right_camera")
            frames = [
                np.asarray(images_dict[key], dtype=np.uint8)
                for key in preferred
                if key in images_dict
            ]
        else:
            frames = [np.asarray(image, dtype=np.uint8) for image in images_dict.values()]

        if not frames:
            frames = [np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)]
        while len(frames) < self._num_real_cameras:
            frames.append(np.zeros_like(frames[0]))
        frames = frames[: self._num_real_cameras]

        raw_state = obs.get("states", obs.get("state", obs.get("joint_state")))
        if raw_state is None:
            raw_state = np.zeros(self._state_dim, dtype=np.float32)
        state = self._fit_state(raw_state)
        if self.embodiment == "robotwin":
            from vla_eval.model_servers.embodiment_adapt import robotwin_state_to_pi

            state = robotwin_state_to_pi(state)

        return {
            "images": {f"image_{index}": frame for index, frame in enumerate(frames)},
            "states": state,
            "task_description": obs.get("task_description", ""),
        }

    def _apply_embodiment_actions(
        self, actions: np.ndarray, phyai_obs_list: list[dict[str, Any]]
    ) -> np.ndarray:
        if self.embodiment != "robotwin":
            return actions
        from vla_eval.model_servers.embodiment_adapt import robotwin_pi_actions_to_env

        out = np.array(actions, copy=True)
        for index, phyai_obs in enumerate(phyai_obs_list):
            out[index] = robotwin_pi_actions_to_env(out[index], phyai_obs["states"])
        return out

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._processor is not None and self._engine is not None

        phyai_obs = self._observation_to_phyai_obs(obs)
        noise = self._resolve_noise([ctx], 1)
        request = self._build_request(phyai_obs, noise=noise)
        raw_actions = self._engine_step_timed(request)
        real_actions = self._apply_embodiment_actions(
            self._decode_actions(raw_actions), [phyai_obs]
        )

        logger.info("real_actions shape: %s", real_actions.shape)
        return {"actions": real_actions[0]}

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        self._load_model()
        assert self._processor is not None and self._engine is not None

        n_real = len(obs_batch)
        if len(ctx_batch) != n_real:
            raise ValueError(
                f"obs_batch/ctx_batch length mismatch: {n_real} vs {len(ctx_batch)}"
            )
        real_obs = [self._observation_to_phyai_obs(obs) for obs in obs_batch]
        phyai_obs_list = self._pad_obs_with_length_dummy(real_obs)
        noise = self._resolve_noise(ctx_batch, len(phyai_obs_list))
        request = self._build_batch_request(phyai_obs_list, noise=noise)
        raw_actions = self._engine_step_timed(request)
        real_actions = self._apply_embodiment_actions(
            self._decode_actions(raw_actions), real_obs
        )

        logger.info(
            "batch real_actions shape: %s (business_B=%d, engine_B=%d, pad_length_dummy=%s)",
            real_actions.shape,
            n_real,
            len(phyai_obs_list),
            self.pad_length_dummy,
        )
        return [{"actions": real_actions[i]} for i in range(n_real)]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(PhyAIModelServer)
