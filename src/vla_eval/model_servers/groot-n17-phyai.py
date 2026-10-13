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
# exclude-newer = "2026-07-31T00:00:00Z"
# ///
"""PhyAI GR00T-N1.7 model server — full-precision or QVLA W4A4 quantization.

Serves GR00T-N1.7 through phyai's engine with support for:

* **FP (default):** omit ``qvla_pack_path`` (or set to ``null``). Engine is
  built with CUDA Graph capture profiles for the configured task suite.
* **QVLA:** set ``qvla_pack_path`` to a pack built via
  ``qvla/scripts/build_groot_pack.py``. Eager-mode engine is built first, then
  W4A4 layer swap; CUDA graphs are recaptured afterwards if requested.
* **Differentiable forward:** set ``use_differentiable_forward=True`` to install
  qvla's autograd patches (linears + norms) for the Fisher/policy-aware
  calibration path. Requires ``use_cuda_graph=False``.
* **Reproducible eval:** set ``reproducible=True`` to enable length-dummy
  padding, ``inference_seed`` (default ``0``), deterministic RNG, and an
  explicit noise mode (``random`` / ``global`` / ``episode``).
  ``"random"`` is fresh OS-entropy noise, independent of ``inference_seed``;
  ``"global"`` is one fixed noise from ``inference_seed``; ``"episode"`` is
  per-sample ``hash(seed, task_id, episode_idx, step)``. ``noise_mode="global"``
  or ``"episode"`` requires ``reproducible=True``.

Length-dummy pad (also required when ``use_cuda_graph=True``) fills unused
engine batch slots with the **longest** suite prompt (by tokenized length) so
every request hits the same max sequence bucket.

CUDA Graphs need a known ``model_suite`` (``libero_goal``, ``simpler_widowx``,
``simpler_google``, …). Its built-in task list is the capture profile.
The checkpoint directory defaults to ``checkpoint_root / model_suite``.
Synthetic capture / dummy images use ``processor.image_target_size``.

Supports vla-eval batch parallel evaluation: set ``max_batch_size > 1`` and run
sharded benchmarks. The framework coalesces concurrent observations into
``predict_batch()``; phyai's GR00T-N1.7 scheduler runs a single GPU forward
pass per batch.
"""

from __future__ import annotations

import hashlib
import logging
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import (
    DeviceConfig,
    EngineConfig,
    RuntimeConfig,
    get_engine_config,
    set_engine_config,
)
from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request
from phyai.utils import load_config

from phyai_utils_tools.models.gr00t import (
    GR00TObservation,
    GR00TProcessedInputs,
    GR00TProcessor,
)

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_01,
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_EULER,
    DimSpec,
)
from vla_eval.types import Action, Observation

from qvla import enable_quantization
from qvla.adapters.groot.differentiable import differentiable_inference_context
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


# Per-suite language instructions used as CUDA Graph capture profiles.
# Texts match vla-eval LIBERO benchmark task language (see
# docs/reproductions/data/xvla-libero). Equivalent tokenized structures are
# deduplicated by the scheduler before capture.
_LIBERO_10_CAPTURE_TASKS: tuple[str, ...] = (
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)
_LIBERO_GOAL_CAPTURE_TASKS: tuple[str, ...] = (
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet",
    "push the plate to the front of the stove",
    "put the cream cheese in the bowl",
    "turn on the stove",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
)
_LIBERO_OBJECT_CAPTURE_TASKS: tuple[str, ...] = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
)
_LIBERO_SPATIAL_CAPTURE_TASKS: tuple[str, ...] = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
)

_LIBERO_90_CAPTURE_TASKS: tuple[str, ...] = (
    "close the top drawer of the cabinet",
    "close the top drawer of the cabinet and put the black bowl on top of it",
    "put the black bowl in the top drawer of the cabinet",
    "put the butter at the back in the top drawer of the cabinet and close it",
    "put the butter at the front in the top drawer of the cabinet and close it",
    "put the chocolate pudding in the top drawer of the cabinet and close it",
    "open the bottom drawer of the cabinet",
    "open the top drawer of the cabinet",
    "open the top drawer of the cabinet and put the bowl in it",
    "put the black bowl on the plate",
    "put the black bowl on top of the cabinet",
    "open the top drawer of the cabinet",
    "put the black bowl at the back on the plate",
    "put the black bowl at the front on the plate",
    "put the middle black bowl on the plate",
    "put the middle black bowl on top of the cabinet",
    "stack the black bowl at the front on the black bowl in the middle",
    "stack the middle black bowl on the back black bowl",
    "put the frying pan on the stove",
    "put the moka pot on the stove",
    "turn on the stove",
    "turn on the stove and put the frying pan on it",
    "close the bottom drawer of the cabinet",
    "close the bottom drawer of the cabinet and open the top drawer",
    "put the black bowl in the bottom drawer of the cabinet",
    "put the black bowl on top of the cabinet",
    "put the wine bottle in the bottom drawer of the cabinet",
    "put the wine bottle on the wine rack",
    "close the top drawer of the cabinet",
    "put the black bowl in the top drawer of the cabinet",
    "put the black bowl on the plate",
    "put the black bowl on top of the cabinet",
    "put the ketchup in the top drawer of the cabinet",
    "close the microwave",
    "put the yellow and white mug to the front of the white mug",
    "open the microwave",
    "put the white bowl on the plate",
    "put the white bowl to the right of the plate",
    "put the right moka pot on the stove",
    "turn off the stove",
    "put the frying pan on the cabinet shelf",
    "put the frying pan on top of the cabinet",
    "put the frying pan under the cabinet shelf",
    "put the white bowl on top of the cabinet",
    "turn on the stove",
    "turn on the stove and put the frying pan on it",
    "pick up the alphabet soup and put it in the basket",
    "pick up the cream cheese box and put it in the basket",
    "pick up the ketchup and put it in the basket",
    "pick up the tomato sauce and put it in the basket",
    "pick up the alphabet soup and put it in the basket",
    "pick up the butter and put it in the basket",
    "pick up the milk and put it in the basket",
    "pick up the orange juice and put it in the basket",
    "pick up the tomato sauce and put it in the basket",
    "pick up the alphabet soup and put it in the tray",
    "pick up the butter and put it in the tray",
    "pick up the cream cheese and put it in the tray",
    "pick up the ketchup and put it in the tray",
    "pick up the tomato sauce and put it in the tray",
    "pick up the black bowl on the left and put it in the tray",
    "pick up the chocolate pudding and put it in the tray",
    "pick up the salad dressing and put it in the tray",
    "stack the left bowl on the right bowl and place them in the tray",
    "stack the right bowl on the left bowl and place them in the tray",
    "put the red mug on the left plate",
    "put the red mug on the right plate",
    "put the white mug on the left plate",
    "put the yellow and white mug on the right plate",
    "put the chocolate pudding to the left of the plate",
    "put the chocolate pudding to the right of the plate",
    "put the red mug on the plate",
    "put the white mug on the plate",
    "pick up the book and place it in the front compartment of the caddy",
    "pick up the book and place it in the left compartment of the caddy",
    "pick up the book and place it in the right compartment of the caddy",
    "pick up the yellow and white mug and place it to the right of the caddy",
    "pick up the book and place it in the back compartment of the caddy",
    "pick up the book and place it in the front compartment of the caddy",
    "pick up the book and place it in the left compartment of the caddy",
    "pick up the book and place it in the right compartment of the caddy",
    "pick up the book and place it in the front compartment of the caddy",
    "pick up the book and place it in the left compartment of the caddy",
    "pick up the book and place it in the right compartment of the caddy",
    "pick up the red mug and place it to the right of the caddy",
    "pick up the white mug and place it to the right of the caddy",
    "pick up the book in the middle and place it on the cabinet shelf",
    "pick up the book on the left and place it on top of the shelf",
    "pick up the book on the right and place it on the cabinet shelf",
    "pick up the book on the right and place it under the cabinet shelf",
)

# SimplerEnv WidowX VM. Strings are the env's get_language_instruction() values.
_SIMPLER_WIDOWX_CAPTURE_TASKS: tuple[str, ...] = (
    "put the spoon on the towel",
    "put carrot on plate",
    "stack the green block on the yellow block",
    "put the eggplant in the yellow basket",
)

# SimplerEnv Google Robot VM and VA. Built from the same object-name cleanup
# the env uses (drop opened/baked/v2/size) plus drawer ids top/middle/bottom.
_SIMPLER_GOOGLE_CAPTURE_TASKS: tuple[str, ...] = (
    "pick pepsi can",
    "pick coke can",
    "pick sprite can",
    "pick fanta can",
    "pick redbull can",
    "pick blue plastic bottle",
    "pick apple",
    "pick orange",
    "pick sponge",
    "pick spoon",
    "pick carrot",
    "pick green cube",
    "pick yellow cube",
    "pick eggplant",
    "move blue plastic bottle near pepsi can",
    "move blue plastic bottle near orange",
    "move pepsi can near blue plastic bottle",
    "move pepsi can near orange",
    "move orange near blue plastic bottle",
    "move orange near pepsi can",
    "move 7up can near apple",
    "move 7up can near sponge",
    "move apple near 7up can",
    "move apple near sponge",
    "move sponge near 7up can",
    "move sponge near apple",
    "move coke can near redbull can",
    "move coke can near apple",
    "move redbull can near coke can",
    "move redbull can near apple",
    "move apple near coke can",
    "move apple near redbull can",
    "move sponge near blue plastic bottle",
    "move blue plastic bottle near sponge",
    "move blue plastic bottle near 7up can",
    "move 7up can near blue plastic bottle",
    "move orange near redbull can",
    "move pepsi can near redbull can",
    "move redbull can near orange",
    "move redbull can near pepsi can",
    "open top drawer",
    "close top drawer",
    "open middle drawer",
    "close middle drawer",
    "open bottom drawer",
    "close bottom drawer",
    "place pepsi can into top drawer",
    "place pepsi can into middle drawer",
    "place pepsi can into bottom drawer",
    "place coke can into top drawer",
    "place coke can into middle drawer",
    "place coke can into bottom drawer",
    "place sprite can into top drawer",
    "place sprite can into middle drawer",
    "place sprite can into bottom drawer",
    "place 7up can into top drawer",
    "place 7up can into middle drawer",
    "place 7up can into bottom drawer",
    "place fanta can into top drawer",
    "place fanta can into middle drawer",
    "place fanta can into bottom drawer",
    "place redbull can into top drawer",
    "place redbull can into middle drawer",
    "place redbull can into bottom drawer",
    "place blue plastic bottle into top drawer",
    "place blue plastic bottle into middle drawer",
    "place blue plastic bottle into bottom drawer",
    "place apple into top drawer",
    "place apple into middle drawer",
    "place apple into bottom drawer",
    "place orange into top drawer",
    "place orange into middle drawer",
    "place orange into bottom drawer",
    "place sponge into top drawer",
    "place sponge into middle drawer",
    "place sponge into bottom drawer",
)

CAPTURE_SUITE_TASKS: dict[str, tuple[str, ...]] = {
    "libero_10": _LIBERO_10_CAPTURE_TASKS,
    "libero_90": _LIBERO_90_CAPTURE_TASKS,
    "libero_goal": _LIBERO_GOAL_CAPTURE_TASKS,
    "libero_object": _LIBERO_OBJECT_CAPTURE_TASKS,
    "libero_spatial": _LIBERO_SPATIAL_CAPTURE_TASKS,
    "simpler_widowx": _SIMPLER_WIDOWX_CAPTURE_TASKS,
    "simpler_google": _SIMPLER_GOOGLE_CAPTURE_TASKS,
}


def _resolve_capture_tasks(model_suite: str) -> tuple[str, ...]:
    key = model_suite.strip().lower()
    if key not in CAPTURE_SUITE_TASKS:
        raise ValueError(
            f"Unknown model_suite={model_suite!r}. "
            f"Choose from {sorted(CAPTURE_SUITE_TASKS)}."
        )
    return CAPTURE_SUITE_TASKS[key]


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
    request: GR00TN17Request,
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


class PhyAIGR00TModelServer(PredictModelServer):
    """GR00T-N1.7 server — FP by default; QVLA when ``qvla_pack_path`` is set."""

    def __init__(
        self,
        checkpoint_root: str,
        model_suite: str,
        noise_mode: str,
        model_kind: str,
        embodiment_tag: str = "LIBERO_PANDA",
        checkpoint_path: str | None = None,
        state_layout: str | None = None,
        device: str = "cuda",
        params_dtype: str = "bfloat16",
        use_cuda_graph: bool = True,
        max_batch_size: int = 1,
        qvla_pack_path: str | None = None,
        strict_pack_match: bool = True,
        use_differentiable_forward: bool = False,
        reproducible: bool = False,
        inference_seed: int = 0,
        invert_gripper: bool = True,
        processor_model_name_or_path: str | None = None,
        trust_remote_code: bool = False,
        online: bool = False,
        n_warmup: int = 3,
        n_timed_dummy: int = 3,
        n_timed_real: int = 3,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)
        # None keeps the embodiment action horizon. SimplerEnv official eval
        # sets 1 so the policy is queried every env step.
        self._chunk_size_override = None if chunk_size is None else int(chunk_size)
        self.model_suite = model_suite.strip().lower()
        self._capture_tasks = _resolve_capture_tasks(self.model_suite)
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else self.checkpoint_root / self.model_suite
        )
        self.state_layout = state_layout.strip().lower() if state_layout else None
        if self.state_layout not in (None, "simpler_widowx", "simpler_google"):
            raise ValueError(
                "state_layout must be simpler_widowx, simpler_google, or omitted, "
                f"got {state_layout!r}."
            )
        self.model_kind = model_kind.strip().lower()
        if not self.model_kind or any(
            c != "_" and not c.isalnum() for c in self.model_kind
        ):
            raise ValueError(
                "model_kind must be a non-empty alphanumeric/underscore slug, "
                f"got {model_kind!r}."
            )
        self.embodiment_tag = embodiment_tag
        self.invert_gripper = bool(invert_gripper)
        self.device = device
        self.params_dtype = _DTYPE_MAP[params_dtype]
        self.use_cuda_graph = bool(use_cuda_graph)
        self.reproducible = bool(reproducible)
        self.noise_mode = _parse_noise_mode(noise_mode)
        if self.noise_mode in ("global", "episode") and not self.reproducible:
            raise ValueError(
                f"noise_mode={self.noise_mode!r} requires reproducible=True."
            )
        self._engine_max_batch_size = max_batch_size
        if self.reproducible:
            if max_batch_size < 2:
                raise ValueError(
                    "reproducible=True requires max_batch_size >= 2 "
                    "(one slot is reserved for the length-dummy pad)."
                )
            self.pad_length_dummy = True
            if type(inference_seed) is not int:
                raise ValueError(
                    "inference_seed must be an int, "
                    f"got {inference_seed!r}."
                )
            self.inference_seed: int | None = inference_seed
            self.deterministic_algorithms = True
            self.max_batch_size = max_batch_size - 1
        elif self.use_cuda_graph:
            # GR00T CUDA graphs require fixed batch size matching capture profiles.
            # Padding is mandatory even without reproducible mode.
            self.pad_length_dummy = True
            self.inference_seed = None
            self.deterministic_algorithms = False
            self.max_batch_size = max_batch_size
        else:
            self.pad_length_dummy = False
            self.inference_seed = None
            self.deterministic_algorithms = False
            self.max_batch_size = max_batch_size
        if self.noise_mode in ("global", "episode") and self.inference_seed is None:
            raise ValueError(
                f"noise_mode={self.noise_mode!r} requires inference_seed "
                "(set reproducible=True)."
            )
        self.qvla_pack_path = (
            Path(qvla_pack_path) if qvla_pack_path else None
        )
        self.strict_pack_match = bool(strict_pack_match)
        self.use_differentiable_forward = bool(use_differentiable_forward)
        logger.info(
            "checkpoint_root=%s model_suite=%s checkpoint_path=%s",
            self.checkpoint_root,
            self.model_suite,
            self.checkpoint_path,
        )
        self.processor_model_name_or_path = processor_model_name_or_path
        self.trust_remote_code = bool(trust_remote_code)
        self.online = bool(online)
        self.n_warmup = n_warmup
        self.n_timed_dummy = n_timed_dummy
        self.n_timed_real = n_timed_real
        self._episode_keys: dict[str, tuple[int, int]] = {}

        self._model_config: GR00TN17Config | None = None
        self._processor: GR00TProcessor | None = None
        self._engine: Engine | None = None
        self._capture_profiles: tuple[GR00TN17Request, ...] | None = None
        # Suite prompt with max tokenized length; set during processor setup
        # when pad_length_dummy is enabled.
        self._pad_language: str | None = None
        self._real_times_ms: list[float] = []

        # Embodiment action chunk (env / decode). DiT noise uses model horizons.
        self._action_horizon: int = 16
        self._model_action_horizon: int = 40
        self._max_action_dim: int = 132

        logger.info(
            "reproducible=%s use_cuda_graph=%s model_suite=%s "
            "pad_length_dummy=%s inference_seed=%s deterministic_algorithms=%s "
            "noise_mode=%s max_batch_size=%d engine_bs=%d embodiment_tag=%s",
            self.reproducible,
            self.use_cuda_graph,
            self.model_suite,
            self.pad_length_dummy,
            self.inference_seed,
            self.deterministic_algorithms,
            self.noise_mode,
            self.max_batch_size,
            self._engine_max_batch_size,
            self.embodiment_tag,
        )

    @property
    def _qvla_enabled(self) -> bool:
        return self.qvla_pack_path is not None

    def _capture_image_hw(self) -> tuple[int, int]:
        """Square H/W from ``processor.image_target_size`` for CUDA-graph capture."""
        assert self._processor is not None
        target = self._processor.image_target_size
        if target is None:
            raise ValueError(
                "processor.image_target_size is None; cannot build synthetic "
                "capture/dummy images. Set image_target_size on the processor."
            )
        h, w = int(target[0]), int(target[1])
        if h != w:
            raise ValueError(
                f"processor.image_target_size must be square for synthetic "
                f"capture/dummy images, got {(h, w)}."
            )
        return h, w

    def _raw_image_hw(self) -> tuple[int, int]:
        """H/W of frames stacked before the GR00T image transform.

        Official ``GoogleFractalEnv`` resizes to ``(H, W) = (256, 320)``.
        The processor then letterboxes non-square frames. Squashing the same
        camera image to the square processor target changes the pixels the
        policy sees, which is enough to zero out the drawer tasks.
        WidowX is already ``(256, 256)`` in ``WidowXBridgeEnv``.
        """
        if self.state_layout == "simpler_google":
            return 256, 320
        return self._capture_image_hw()

    def _resize_video_to_target(self, img: np.ndarray) -> np.ndarray:
        """Resize (T, H, W, C) uint8 frames to the embodiment's raw H/W.

        Real env frames arrive at native resolution while length-dummy frames
        use the same raw H/W; without normalizing, batching the two fails
        with a shape mismatch. Google frames match ``GoogleFractalEnv``
        (256x320). Other embodiments stay on the square processor target.
        """
        th, tw = self._raw_image_hw()
        if img.shape[1] == th and img.shape[2] == tw:
            return img
        import cv2

        frames = [
            cv2.resize(img[t], (tw, th), interpolation=cv2.INTER_LINEAR)
            for t in range(img.shape[0])
        ]
        return np.stack(frames, axis=0)

    def _seed_rng(self) -> None:
        seed = int(self.inference_seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("RNG seeded (seed=%s).", seed)

    def _enable_deterministic_algorithms(self) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        logger.info("Deterministic algorithms on.")

    def _sample_noise_from_seed(self, seed: int) -> torch.Tensor:
        """Sample flow-matching noise matching the action-head CUDA Graph layout.

        Must use the **model** action-head horizon/dim (e.g. 40×132), not the
        embodiment/processor action chunk (e.g. LIBERO 16). Capture profiles
        fix noise to ``(B, action_head.action_horizon, action_head.action_dim)``.
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) % (2**31))
        return torch.randn(
            1,
            self._model_action_horizon,
            self._max_action_dim,
            generator=gen,
            dtype=torch.float32,
        )

    def _sample_noise_from_unseed(self, batch_size: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int.from_bytes(os.urandom(8), "little"))
        return torch.randn(
            batch_size,
            self._model_action_horizon,
            self._max_action_dim,
            generator=gen,
            dtype=torch.float32,
        )

    @staticmethod
    def _episode_noise_seed(
        base_seed: int, task_id: int, episode_idx: int, step: int
    ) -> int:
        payload = f"{int(base_seed)}:{int(task_id)}:{int(episode_idx)}:{int(step)}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % (2**31)

    def _make_noise(self, batch_size: int) -> torch.Tensor:
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

    # ------------------------------------------------------------------
    # Observation → GR00TObservation conversion
    # ------------------------------------------------------------------

    def _observation_to_groot_obs(self, obs: Observation) -> dict[str, Any]:
        """Convert a vla-eval Observation to the raw dict format for batching.

        Returns a dict with video/state/language suitable for building a
        GR00TObservation once batched.
        """
        assert self._processor is not None
        config = self._processor.modality_config

        images_dict = obs.get("images", {})
        img_list = list(images_dict.values()) if isinstance(images_dict, dict) else []

        video_keys = config["video"].modality_keys
        video: dict[str, np.ndarray] = {}
        t_video = len(config["video"].delta_indices)
        for idx, vk in enumerate(video_keys):
            if idx < len(img_list):
                img = np.asarray(img_list[idx], dtype=np.uint8)
                if img.ndim == 3:
                    img = img[np.newaxis, ...]  # (T=1, H, W, C)
                if img.shape[0] < t_video:
                    img = np.repeat(img, t_video, axis=0)[:t_video]
                img = self._resize_video_to_target(img)
            else:
                th, tw = self._raw_image_hw()
                img = np.zeros((t_video, th, tw, 3), dtype=np.uint8)
            video[vk] = img

        state_keys = config["state"].modality_keys
        raw_state_flat = obs.get("states", obs.get("state"))
        state: dict[str, np.ndarray] = {}
        t_state = len(config["state"].delta_indices)
        if raw_state_flat is not None:
            state_arr = np.asarray(raw_state_flat, dtype=np.float32).flatten()
            state_arr = self._adapt_flat_state(state_arr)
        else:
            state_arr = np.zeros(0, dtype=np.float32)
        offset = 0
        for sk in state_keys:
            dim = int(self._processor.norm_params[self._processor.embodiment_tag]["state"][sk]["dim"])
            if offset + dim <= len(state_arr):
                val = state_arr[offset: offset + dim]
            else:
                val = np.zeros(dim, dtype=np.float32)
            state[sk] = np.tile(val, (t_state, 1))  # (T, D)
            offset += dim

        language_key = config["language"].modality_keys[0]
        t_lang = len(config["language"].delta_indices)
        task_desc = str(obs.get("task_description", ""))
        language: dict[str, list[str]] = {
            language_key: [task_desc] * t_lang,
        }

        return {"video": video, "state": state, "language": language}

    def _adapt_flat_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_layout is None:
            return state
        from vla_eval.model_servers.embodiment_adapt import (
            simpler_google_state,
            simpler_widowx_state,
        )

        if self.state_layout == "simpler_widowx":
            return simpler_widowx_state(state)
        return simpler_google_state(state)

    def _batch_groot_obs(
        self, obs_dicts: list[dict[str, Any]]
    ) -> GR00TObservation:
        """Stack per-sample observation dicts into a batched GR00TObservation."""
        assert self._processor is not None
        config = self._processor.modality_config

        video_keys = config["video"].modality_keys
        video: dict[str, np.ndarray] = {}
        for vk in video_keys:
            video[vk] = np.stack([d["video"][vk] for d in obs_dicts], axis=0)

        state_keys = config["state"].modality_keys
        state: dict[str, np.ndarray] = {}
        for sk in state_keys:
            state[sk] = np.stack([d["state"][sk] for d in obs_dicts], axis=0)

        language_key = config["language"].modality_keys[0]
        language: dict[str, list[list[str]]] = {
            language_key: [d["language"][language_key] for d in obs_dicts],
        }

        return GR00TObservation(video=video, state=state, language=language)

    def _obs_language(self, obs_dict: dict[str, Any]) -> str:
        assert self._processor is not None
        language_key = self._processor.modality_config["language"].modality_keys[0]
        texts = obs_dict["language"][language_key]
        if not texts:
            raise ValueError("observation language list is empty.")
        language = str(texts[0])
        if any(str(t) != language for t in texts):
            raise ValueError(
                f"per-timestep language mismatch within one observation: {texts!r}."
            )
        if not language:
            raise ValueError("observation language is an empty string.")
        return language

    def _make_dummy_groot_obs_dict(self, *, language: str) -> dict[str, Any]:
        """Build a dummy observation with the given language prompt."""
        assert self._processor is not None
        config = self._processor.modality_config

        video_keys = config["video"].modality_keys
        t_video = len(config["video"].delta_indices)
        image_h, image_w = self._raw_image_hw()
        video: dict[str, np.ndarray] = {}
        for vk in video_keys:
            video[vk] = np.random.randint(
                0, 255,
                (t_video, image_h, image_w, 3),
                dtype=np.uint8,
            )

        state_keys = config["state"].modality_keys
        t_state = len(config["state"].delta_indices)
        state: dict[str, np.ndarray] = {}
        for sk in state_keys:
            dim = int(self._processor.norm_params[self._processor.embodiment_tag]["state"][sk]["dim"])
            state[sk] = np.random.randn(t_state, dim).astype(np.float32)

        language_key = config["language"].modality_keys[0]
        t_lang = len(config["language"].delta_indices)
        language_dict: dict[str, list[str]] = {
            language_key: [language] * t_lang,
        }

        return {"video": video, "state": state, "language": language_dict}

    def _tokenized_language_length(self, language: str) -> int:
        """Valid-token count for one synthetic obs with ``language``."""
        assert self._processor is not None
        obs = self._batch_groot_obs(
            [self._make_dummy_groot_obs_dict(language=language)]
        )
        prepared = self._processor.process_observation(obs)
        attention_mask = prepared.tensors["attention_mask"]
        if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
            raise ValueError(
                "expected attention_mask shape (1, S) when measuring pad "
                f"language length, got {tuple(attention_mask.shape)}."
            )
        length = int(attention_mask.sum(dim=-1).item())
        if length <= 0:
            raise ValueError(
                f"tokenized language {language!r} has no valid tokens."
            )
        return length

    def _resolve_pad_language(self) -> str:
        """Pick the capture-suite prompt with the longest tokenized length."""
        if not self._capture_tasks:
            raise ValueError(
                "cannot resolve pad language without a capture suite."
            )
        best_language: str | None = None
        best_length = -1
        for task in self._capture_tasks:
            length = self._tokenized_language_length(task)
            if length > best_length:
                best_length = length
                best_language = task
        assert best_language is not None
        logger.info(
            "length-dummy pad language (max tokenized length=%d): %r",
            best_length,
            best_language,
        )
        return best_language

    def _pad_obs_with_length_dummy(
        self, obs_dicts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Pad to ``_engine_max_batch_size`` with longest-suite-prompt dummies."""
        if not self.pad_length_dummy:
            return obs_dicts
        n = len(obs_dicts)
        if n > self.max_batch_size:
            raise ValueError(
                f"business batch size {n} exceeds max_batch_size={self.max_batch_size}"
            )
        if self._pad_language is None:
            raise RuntimeError(
                "pad_length_dummy=True but suite-max pad language was not "
                "resolved; processor setup must run first."
            )
        suite_languages = set(self._capture_tasks)
        for obs_dict in obs_dicts:
            language = self._obs_language(obs_dict)
            if language not in suite_languages:
                raise ValueError(
                    f"observation language {language!r} is not in capture "
                    f"suite {self.model_suite!r}; refusing to pad so a "
                    "longer-than-suite prompt cannot silently change the "
                    f"sequence bucket. Known prompts: {sorted(suite_languages)}."
                )
        n_pad = self._engine_max_batch_size - n
        return [
            *obs_dicts,
            *[
                self._make_dummy_groot_obs_dict(language=self._pad_language)
                for _ in range(n_pad)
            ],
        ]

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(
        self,
        observation: GR00TObservation,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[GR00TN17Request, GR00TProcessedInputs]:
        """Process observation and build engine request."""
        assert self._processor is not None
        prepared = self._processor.process_observation(observation)
        tensors = {
            key: value.to(device=self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in prepared.tensors.items()
        }
        if noise is not None:
            noise = noise.to(device=self.device, dtype=self.params_dtype)
        request = GR00TN17Request(tensors=tensors, noise=noise)
        return request, prepared

    # ------------------------------------------------------------------
    # Engine step
    # ------------------------------------------------------------------

    def _step(self, request: GR00TN17Request) -> torch.Tensor:
        """One inference step — engine.step or differentiable context."""
        assert self._engine is not None
        if not self.use_differentiable_forward:
            return self._engine.step(request)

        sched = getattr(self._engine.entry, "scheduler", None)
        if sched is None:
            raise RuntimeError(
                "use_differentiable_forward=True but engine has no scheduler."
            )
        reset_step_counters(self._engine.entry.model)
        with torch.enable_grad():
            with differentiable_inference_context(sched):
                actions = self._engine.step(request)
        return actions.detach()

    def _engine_step_timed(self, request: GR00TN17Request) -> torch.Tensor:
        assert self._engine is not None
        if self.n_timed_real <= 0 or len(self._real_times_ms) >= self.n_timed_real:
            return self._step(request)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        raw_actions = self._step(request)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        self._real_times_ms.append(elapsed_ms)
        batch_size = int(next(iter(request.tensors.values())).shape[0])
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

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_actions(
        self,
        normalized_action: torch.Tensor,
        raw_state: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        """Decode normalized engine output to physical actions."""
        assert self._processor is not None
        return self._processor.decode_action(
            normalized_action.detach(), raw_state=raw_state
        )

    # ------------------------------------------------------------------
    # Engine construction
    # ------------------------------------------------------------------

    def _build_capture_profiles(self) -> tuple[GR00TN17Request, ...]:
        """Build CUDA graph capture profiles from ``model_suite``."""
        assert self._processor is not None
        if not self._capture_tasks:
            raise ValueError(
                "use_cuda_graph=True requires a known model_suite "
                f"(one of {sorted(CAPTURE_SUITE_TASKS)})."
            )
        profiles: list[GR00TN17Request] = []
        config = self._processor.modality_config
        image_h, image_w = self._raw_image_hw()
        for task in self._capture_tasks:
            video = {
                key: np.zeros(
                    (
                        self._engine_max_batch_size,
                        len(config["video"].delta_indices),
                        image_h,
                        image_w,
                        3,
                    ),
                    dtype=np.uint8,
                )
                for key in config["video"].modality_keys
            }
            state = {
                key: np.zeros(
                    (
                        self._engine_max_batch_size,
                        len(config["state"].delta_indices),
                        int(self._processor.norm_params[self._processor.embodiment_tag]["state"][key]["dim"]),
                    ),
                    dtype=np.float32,
                )
                for key in config["state"].modality_keys
            }
            language = {
                key: [
                    [task for _ in config["language"].delta_indices]
                    for _ in range(self._engine_max_batch_size)
                ]
                for key in config["language"].modality_keys
            }
            obs = GR00TObservation(video=video, state=state, language=language)
            prepared = self._processor.process_observation(obs)
            profiles.append(GR00TN17Request(tensors=prepared.tensors))
        logger.info(
            "Built %d capture profile(s) from model_suite=%s "
            "(batch_size=%d, image_hw=%dx%d).",
            len(profiles),
            self.model_suite,
            self._engine_max_batch_size,
            image_h,
            image_w,
        )
        return tuple(profiles)

    def _build_engine(
        self,
        *,
        use_cuda_graph: bool,
        capture_profiles: tuple[GR00TN17Request, ...] | None = None,
    ) -> Engine:
        assert self._processor is not None
        plugin_args = GR00TN17Args(
            checkpoint_dir=self.checkpoint_path,
            max_batch_size=self._engine_max_batch_size,
            capture_profiles=capture_profiles if capture_profiles else (),
        )
        return Engine(
            EngineArgs(
                plugin="gr00t_n17",
                plugin_args=plugin_args,
                config=EngineConfig(
                    device=DeviceConfig(
                        target=self.device, params_dtype=self.params_dtype
                    ),
                    runtime=RuntimeConfig(use_cuda_graph=use_cuda_graph),
                ),
            )
        )

    def _setup_processor(self) -> None:
        self._model_config = load_config(self.checkpoint_path, GR00TN17Config)
        loading_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": not self.online,
        }
        model_name = (
            self.processor_model_name_or_path
            or self._model_config.backbone.model_name
        )
        self._processor = GR00TProcessor.from_pretrained(
            self.checkpoint_path,
            embodiment_tag=self.embodiment_tag,
            model_name=model_name,
            transformers_loading_kwargs=loading_kwargs,
        )
        action_cfg = self._processor.modality_config["action"]
        self._action_horizon = len(action_cfg.delta_indices)
        self._model_action_horizon = int(self._model_config.action_head.action_horizon)
        self._max_action_dim = int(self._model_config.action_head.max_action_dim)
        processor_action_dim = int(self._processor.max_action_dim)
        if processor_action_dim != self._max_action_dim:
            raise ValueError(
                "processor.max_action_dim does not match model action_head: "
                f"{processor_action_dim} vs {self._max_action_dim}."
            )
        self.chunk_size = (
            self._chunk_size_override
            if self._chunk_size_override is not None
            else self._action_horizon
        )
        logger.info(
            "GR00TProcessor loaded: embodiment=%s env_action_horizon=%d "
            "executed_chunk_size=%d model_action_horizon=%d max_action_dim=%d "
            "video_keys=%s state_keys=%s",
            self.embodiment_tag,
            self._action_horizon,
            self.chunk_size,
            self._model_action_horizon,
            self._max_action_dim,
            self._processor.modality_config["video"].modality_keys,
            self._processor.modality_config["state"].modality_keys,
        )
        if self.pad_length_dummy:
            self._pad_language = self._resolve_pad_language()

    @staticmethod
    def _clear_runner_graphs(runner: Any) -> None:
        graphs = getattr(runner, "graphs", None)
        if isinstance(graphs, dict):
            graphs.clear()
        if getattr(runner, "graph", None) is not None:
            runner.graph = None
        warmed = getattr(runner, "_warmed_graph_keys", None)
        if isinstance(warmed, set):
            warmed.clear()
        vision_states = getattr(runner, "_vision_graph_states", None)
        if isinstance(vision_states, dict):
            vision_states.clear()

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
        from dataclasses import replace

        eng = get_engine_config()
        if eng.runtime.use_cuda_graph:
            logger.warning(
                "EngineConfig.use_cuda_graph=True after Engine init. "
                "Forcing eager mode before QVLA layer swap."
            )
            set_engine_config(
                eng.replace(runtime=replace(eng.runtime, use_cuda_graph=False))
            )

        sched = getattr(engine.entry, "scheduler", None)
        if sched is None:
            return

        if getattr(sched, "use_cuda_graph", False):
            logger.warning(
                "Disabling scheduler CUDA graphs before QVLA swap."
            )
            sched.use_cuda_graph = False

        for runner_name in ("backbone_runner", "action_head_runner"):
            runner = getattr(sched, runner_name, None)
            if runner is None:
                continue
            if getattr(runner, "use_cuda_graph", False):
                logger.warning(
                    "Disabling CUDA graphs on %s before QVLA swap.", runner_name
                )
                runner.use_cuda_graph = False
            cls._clear_runner_graphs(runner)

    @classmethod
    def _recapture_cuda_graphs_after_quant(
        cls, engine: Engine, capture_profiles: tuple[GR00TN17Request, ...]
    ) -> None:
        sched = getattr(engine.entry, "scheduler", None)
        if sched is None:
            return

        from dataclasses import replace

        eng = get_engine_config()
        set_engine_config(
            eng.replace(runtime=replace(eng.runtime, use_cuda_graph=True))
        )

        active_runners: list[str] = []
        for runner_name in ("backbone_runner", "action_head_runner"):
            runner = getattr(sched, runner_name, None)
            cls._clear_runner_graphs(runner)
            if cls._set_runner_cuda_graph_flag(runner, enabled=True):
                active_runners.append(runner_name)

        # GR00T-N1.7 requires both runners; scheduler.capture_profiles() also
        # early-returns on sched.use_cuda_graph=False (set at Engine init when
        # QVLA builds with use_cuda_graph=False), so sync that flag here.
        both_active = len(active_runners) == 2
        if not both_active:
            raise RuntimeError(
                "use_cuda_graph=True requires both backbone_runner and "
                "action_head_runner to support CUDA Graph capture after QVLA "
                f"swap (active={active_runners or 'none'})."
            )

        sched.use_cuda_graph = True
        logger.info(
            "Recapturing CUDA graphs after QVLA swap on: %s",
            ", ".join(active_runners),
        )
        warmup_fp4_cuda_tensors(get_engine_config().device.target)
        sched.setup(capture_profiles=capture_profiles)

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
        """Validate differentiable forward preconditions (patches applied per-step)."""
        assert self._engine is not None
        sched = getattr(self._engine.entry, "scheduler", None)
        if sched is None:
            raise RuntimeError(
                "use_differentiable_forward=True but engine has no scheduler."
            )
        model = getattr(sched, "model", None)
        if model is None:
            raise RuntimeError(
                "use_differentiable_forward=True but scheduler has no model."
            )
        backbone = getattr(model, "backbone", None)
        action_head = getattr(model, "action_head", None)
        if backbone is None or action_head is None:
            raise RuntimeError(
                "use_differentiable_forward=True but model is missing "
                "backbone/action_head."
            )
        logger.info(
            "Differentiable forward enabled: autograd patches will be applied "
            "per-step via differentiable_inference_context."
        )

    def _warmup_engine(self) -> None:
        """Warm up engine with dummy requests and optionally benchmark latency."""
        assert self._engine is not None and self._processor is not None

        # CUDA Graph mode: warmup must reuse a captured profile, otherwise
        # backbone raises CudaGraphError on unmatched input structure.
        if self.use_cuda_graph:
            if not self._capture_profiles:
                raise RuntimeError(
                    "use_cuda_graph=True but no capture profiles were stored "
                    "for warmup."
                )
            prepared = self._capture_profiles[0]
            tensors = {
                key: value.to(device=self.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in prepared.tensors.items()
            }
            dummy_request = GR00TN17Request(tensors=tensors, noise=prepared.noise)
        else:
            if not self._capture_tasks:
                raise RuntimeError(
                    "Eager warmup requires a known model_suite "
                    f"(one of {sorted(CAPTURE_SUITE_TASKS)})."
                )
            if self.pad_length_dummy:
                if self._pad_language is None:
                    raise RuntimeError(
                        "pad_length_dummy=True but suite-max pad language was "
                        "not resolved before eager warmup."
                    )
                warmup_language = self._pad_language
            else:
                warmup_language = self._capture_tasks[0]
            dummy_obs_dicts = [
                self._make_dummy_groot_obs_dict(language=warmup_language)
                for _ in range(self._engine_max_batch_size)
            ]
            dummy_obs = self._batch_groot_obs(dummy_obs_dicts)
            dummy_request, _ = self._build_request(dummy_obs)

        if self.use_differentiable_forward:
            mode = "differentiable"
        elif self._qvla_enabled:
            mode = "QVLA"
        else:
            mode = "FP"
        logger.info(
            "Warming up (engine_batch_size=%d, business_max=%d, "
            "pad_length_dummy=%s, mode=%s, cuda_graph=%s)...",
            self._engine_max_batch_size,
            self.max_batch_size,
            self.pad_length_dummy,
            mode,
            self.use_cuda_graph,
        )
        latency = _benchmark_engine_step(
            self._step,
            dummy_request,
            n_warmup=self.n_warmup,
            n_timed_dummy=self.n_timed_dummy,
        )
        if latency is not None:
            logger.info("%s warm-up complete. step latency: %s", mode, latency)
        else:
            logger.info("%s warm-up complete.", mode)

    def _load_model(self) -> None:
        if self._engine is not None:
            return

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

        capture_profiles: tuple[GR00TN17Request, ...] | None = None
        if self.use_cuda_graph:
            capture_profiles = self._build_capture_profiles()
            self._capture_profiles = capture_profiles

        if self._qvla_enabled:
            self._engine = self._build_engine(use_cuda_graph=False)
            self._force_eager_before_quant(self._engine)
            self._apply_qvla()
            if self.use_cuda_graph:
                assert capture_profiles is not None
                self._recapture_cuda_graphs_after_quant(
                    self._engine, capture_profiles
                )
            else:
                sched = getattr(self._engine.entry, "scheduler", None)
                if sched is not None:
                    sched.setup()
            reset_step_counters(self._engine.entry.model)
        else:
            self._engine = self._build_engine(
                use_cuda_graph=self.use_cuda_graph,
                capture_profiles=capture_profiles,
            )

        if self.use_differentiable_forward:
            self._apply_differentiable_forward()

        self._warmup_engine()

    # ------------------------------------------------------------------
    # ModelServer interface
    # ------------------------------------------------------------------

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"send_wrist_image": True, "send_state": True}
        if self.state_layout == "simpler_widowx":
            params.update(
                success_mode="accumulate",
                gripper_mode="binary",
                deterministic_episodes=True,
                max_episode_steps=300,
            )
        elif self.state_layout == "simpler_google":
            params.update(
                success_mode="accumulate",
                gripper_mode="sticky",
                deterministic_episodes=True,
                max_episode_steps=300,
            )
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:
        # Match groot.py (N1.6 LIBERO): structured dims so orchestrator can warn
        # on gripper/rotation mismatches. invert_gripper emits LIBERO's
        # binary_close_positive convention.
        gripper = GRIPPER_CLOSE_POS if self.invert_gripper else GRIPPER_01
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": gripper,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def _postprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        """Env-facing action post-processing after decode.

        Currently maps model gripper [0,1] (0=close) to LIBERO [-1,1]
        (-1=open) when ``invert_gripper`` is set — same as ``groot.py``.
        Without it, LIBERO's ``gripper < 0`` threshold treats every [0,1]
        value as non-negative and the gripper stays closed.
        """
        if not self.invert_gripper:
            return actions
        out = np.asarray(actions, dtype=np.float32).copy()
        out[..., -1] = 1.0 - 2.0 * out[..., -1]
        return out

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._processor is not None and self._engine is not None

        obs_dict = self._observation_to_groot_obs(obs)
        obs_dicts = self._pad_obs_with_length_dummy([obs_dict])
        batch_obs = self._batch_groot_obs(obs_dicts)

        noise = self._resolve_noise([ctx], len(obs_dicts))
        request, prepared = self._build_request(batch_obs, noise=noise)
        normalized_action = self._engine_step_timed(request)

        action_dict = self._decode_actions(normalized_action, raw_state=prepared.raw_state)
        actions = self._action_dict_to_array(action_dict, n_real=1)
        actions[0] = self._postprocess_actions(actions[0])
        logger.info("predict: actions shape=%s", actions[0].shape)
        return {"actions": actions[0]}

    def predict_batch(
        self, obs_batch: list[Observation], ctx_batch: list[SessionContext]
    ) -> list[Action]:
        self._load_model()
        assert self._processor is not None and self._engine is not None

        n_real = len(obs_batch)
        if len(ctx_batch) != n_real:
            raise ValueError(
                f"obs_batch/ctx_batch length mismatch: {n_real} vs {len(ctx_batch)}"
            )

        obs_dicts = [self._observation_to_groot_obs(obs) for obs in obs_batch]
        obs_dicts_padded = self._pad_obs_with_length_dummy(obs_dicts)
        batch_obs = self._batch_groot_obs(obs_dicts_padded)

        noise = self._resolve_noise(ctx_batch, len(obs_dicts_padded))
        request, prepared = self._build_request(batch_obs, noise=noise)
        normalized_action = self._engine_step_timed(request)

        action_dict = self._decode_actions(normalized_action, raw_state=prepared.raw_state)
        actions = [
            self._postprocess_actions(a)
            for a in self._action_dict_to_array(action_dict, n_real=n_real)
        ]

        logger.info(
            "batch: n_real=%d, engine_B=%d, pad_length_dummy=%s",
            n_real,
            len(obs_dicts_padded),
            self.pad_length_dummy,
        )
        return [{"actions": actions[i]} for i in range(n_real)]

    def _action_dict_to_array(
        self, action_dict: dict[str, np.ndarray], n_real: int
    ) -> list[np.ndarray]:
        """Convert decode_action output dict to per-sample action arrays.

        GR00T's decode_action returns dict[str, np.ndarray] where each value
        has shape (B, action_horizon, action_dim_for_key). We concatenate all
        action keys along the last dimension and return only the real samples.
        """
        assert self._processor is not None
        action_cfg = self._processor.modality_config["action"]
        ordered_keys = [f"action.{k}" for k in action_cfg.modality_keys]
        parts = [action_dict[k] for k in ordered_keys if k in action_dict]
        if not parts:
            raise RuntimeError(
                f"decode_action returned no matching keys. "
                f"Expected keys: {ordered_keys}, got: {sorted(action_dict.keys())}"
            )
        concatenated = np.concatenate(parts, axis=-1)  # (B, H, D_total)
        return [concatenated[i] for i in range(n_real)]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(PhyAIGR00TModelServer)
