"""Legacy LIBERO demo-replay calibration export.

Prefer :mod:`vla_eval.calibration.rollout` for FP-policy rollouts across
libero / calvin / robocasa / robotwin / simpler. This module only replays
LIBERO demonstration HDF5s.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.calibration.format import (
    observation_to_calibration_batch,
    save_calibration_dataset,
)
from vla_eval.calibration.sampling import (
    allocate_phase_quotas,
    allocate_task_quotas,
    candidate_indices_for_trajectory,
)

logger = logging.getLogger(__name__)

NUM_EXPORT_TASKS = 10


@dataclass(frozen=True)
class LiberoExportConfig:
    dataset_dir: str | Path
    suite: str = "libero_object"
    seed: int = 7
    env_seed: int | None = None
    max_episodes_per_task: int = 50
    max_tasks: int | None = None
    state_dim: int = 8
    # Replan-aligned grid (matches common pi0.5 chunk_size) + small temporal jitter.
    chunk_size: int = 50
    jitter: int = 2
    end_margin: int = 10


def observation_to_pi05_batch(obs: dict, *, state_dim: int = 8) -> dict:
    """Map harness LIBERO obs → legacy two-cam ``image`` / ``wrist_image`` layout.

    Kept for the demo-replay path so existing LIBERO calib consumers stay stable.
    FP-rollout export should use :func:`observation_to_calibration_batch` instead.
    """
    batch = observation_to_calibration_batch(obs, state_dim=state_dim)
    img_list = list(batch["images"].values())
    image = (
        np.asarray(img_list[0], dtype=np.uint8)
        if len(img_list) > 0
        else np.zeros((256, 256, 3), dtype=np.uint8)
    )
    wrist_image = (
        np.asarray(img_list[1], dtype=np.uint8)
        if len(img_list) > 1
        else np.zeros_like(image)
    )
    return {
        "images": {"image": image, "wrist_image": wrist_image},
        "states": batch["states"],
        "task_description": batch["task_description"],
    }


def _list_demo_names(demo_path: Path, max_episodes: int) -> list[str]:
    import h5py

    with h5py.File(demo_path, "r") as demo_file:
        if "data" not in demo_file:
            raise ValueError(f"Missing 'data' group in {demo_path}")
        demo_names = list(demo_file["data"].keys())
    if not demo_names:
        raise ValueError(f"No demonstrations found in {demo_path}")
    if any(not name.startswith("demo_") or not name[5:].isdigit() for name in demo_names):
        raise ValueError(f"Invalid demonstration name in {demo_path}; expected demo_<number>")
    demo_names.sort(key=lambda name: int(name[5:]))
    return demo_names[:max_episodes]


def _load_demo_states(demo_path: Path, demo_name: str) -> np.ndarray:
    import h5py

    with h5py.File(demo_path, "r") as demo_file:
        demo = demo_file[f"data/{demo_name}"]
        if "states" not in demo:
            raise ValueError(f"Missing states in {demo_path}:data/{demo_name}")
        states = np.asarray(demo["states"])
    if states.ndim != 2 or len(states) < 2:
        raise ValueError(
            f"Expected at least two MuJoCo states in {demo_path}:data/{demo_name}, "
            f"got shape {states.shape}"
        )
    return states


def _select_picks_for_task(
    *,
    demo_path: Path,
    demo_names: list[str],
    phase_quotas: tuple[int, int, int],
    cfg: LiberoExportConfig,
    rng: random.Random,
) -> list[tuple[str, int]]:
    """Grow candidate pools demo-by-demo, then stratified sample without replacement."""
    pools: list[list[tuple[str, int]]] = [[], [], []]
    order = list(demo_names)
    rng.shuffle(order)

    for demo_name in order:
        if all(len(pools[p]) >= phase_quotas[p] for p in range(3)):
            break
        states = _load_demo_states(demo_path, demo_name)
        for t, phase in candidate_indices_for_trajectory(
            len(states),
            chunk_size=cfg.chunk_size,
            jitter=cfg.jitter,
            end_margin=cfg.end_margin,
            rng=rng,
        ):
            pools[phase].append((demo_name, t))

    for phase, need in enumerate(phase_quotas):
        if len(pools[phase]) < need:
            raise ValueError(
                f"Not enough phase-{phase} candidates in {demo_path}: "
                f"need {need}, have {len(pools[phase])} "
                f"(chunk_size={cfg.chunk_size}, end_margin={cfg.end_margin}, "
                f"demos={len(demo_names)})."
            )

    picks: list[tuple[str, int]] = []
    for phase, need in enumerate(phase_quotas):
        picks.extend(rng.sample(pools[phase], need))
    return picks


def collect_libero_batches(cfg: LiberoExportConfig, num_samples: int) -> list[dict]:
    """Sample 10 tasks with replan-grid + phase-balanced teleport observations."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if cfg.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if cfg.jitter < 0:
        raise ValueError("jitter must be non-negative.")
    if cfg.end_margin < 0:
        raise ValueError("end_margin must be non-negative.")

    dataset_dir = Path(cfg.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"LIBERO dataset directory does not exist: {dataset_dir}")

    bench = LIBEROBenchmark(
        suite=cfg.suite,
        seed=cfg.seed,
        env_seed=cfg.env_seed,
        num_steps_wait=0,
        send_wrist_image=True,
        send_state=True,
    )
    batches: list[dict] = []
    try:
        tasks = bench.get_tasks()
        if cfg.max_tasks is not None:
            tasks = tasks[: cfg.max_tasks]
        if len(tasks) < NUM_EXPORT_TASKS:
            raise RuntimeError(
                f"Demo replay export requires at least {NUM_EXPORT_TASKS} tasks, but suite "
                f"{cfg.suite!r} provided {len(tasks)}."
            )

        rng = random.Random(cfg.seed)
        selected_tasks = rng.sample(tasks, NUM_EXPORT_TASKS)
        task_quotas = allocate_task_quotas(num_samples, NUM_EXPORT_TASKS)

        state_cache: dict[tuple[str, str], np.ndarray] = {}

        for task, task_quota in zip(selected_tasks, task_quotas):
            if task_quota == 0:
                continue
            task_obj = task["task_obj"]
            demo_path = dataset_dir / task_obj.problem_folder / f"{task_obj.name}_demo.hdf5"
            if not demo_path.is_file():
                raise FileNotFoundError(f"LIBERO demonstration file does not exist: {demo_path}")

            demo_names = _list_demo_names(demo_path, cfg.max_episodes_per_task)
            phase_quotas = allocate_phase_quotas(task_quota)
            picks = _select_picks_for_task(
                demo_path=demo_path,
                demo_names=demo_names,
                phase_quotas=phase_quotas,
                cfg=cfg,
                rng=rng,
            )

            replay_task = dict(task)
            for demo_name, state_idx in picks:
                cache_key = (str(demo_path), demo_name)
                if cache_key not in state_cache:
                    state_cache[cache_key] = _load_demo_states(demo_path, demo_name)
                states = state_cache[cache_key]
                raw_obs = bench.reset_to_state(replay_task, states[state_idx])
                batches.append(
                    observation_to_pi05_batch(
                        bench.make_obs(raw_obs, replay_task),
                        state_dim=cfg.state_dim,
                    )
                )
            logger.info(
                "Collected %d/%d calibration samples",
                len(batches),
                num_samples,
            )
    finally:
        bench.cleanup()

    if len(batches) != num_samples:
        raise RuntimeError(
            f"Expected exactly {num_samples} calibration samples, got {len(batches)}."
        )
    return batches


def export_libero_dataset(
    output: str | Path,
    *,
    num_samples: int = 30,
    config: LiberoExportConfig,
) -> Path:
    """Export replayed LIBERO demo observations to a calibration ``.npz``."""
    batches = collect_libero_batches(config, num_samples)
    return save_calibration_dataset(
        output,
        batches,
        meta={
            "source": "libero",
            "exporter": "vla-eval",
            "sampling": (
                "10_tasks_phase_balanced_chunk_grid_jitter_state_teleport"
            ),
            "suite": config.suite,
            "seed": config.seed,
            "env_seed": config.env_seed if config.env_seed is not None else config.seed,
            "chunk_size": config.chunk_size,
            "jitter": config.jitter,
            "end_margin": config.end_margin,
        },
    )


__all__ = [
    "NUM_EXPORT_TASKS",
    "LiberoExportConfig",
    "collect_libero_batches",
    "export_libero_dataset",
    "observation_to_pi05_batch",
]
