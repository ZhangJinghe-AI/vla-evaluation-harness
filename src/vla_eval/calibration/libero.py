"""
Version-1 ``.npz`` calibration dataset format (QVLA compatible).
Replay LIBERO demonstrations and sample observations across each trajectory.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark

logger = logging.getLogger(__name__)

CALIBRATION_FILE_VERSION = 1
Phase = Literal[0, 1, 2]  # early / mid / late
NUM_EXPORT_TASKS = 10


def _batches_to_arrays(batches: list[dict]) -> dict[str, Any]:
    if not batches:
        raise ValueError("Cannot save an empty calibration dataset.")
    images = []
    wrists = []
    states = []
    tasks: list[str] = []
    for batch in batches:
        imgs = batch.get("images", {})
        images.append(np.asarray(imgs["image"], dtype=np.uint8))
        wrists.append(np.asarray(imgs["wrist_image"], dtype=np.uint8))
        states.append(np.asarray(batch["states"], dtype=np.float32))
        tasks.append(str(batch.get("task_description", "")))
    return {
        "images": np.stack(images, axis=0),
        "wrist_images": np.stack(wrists, axis=0),
        "states": np.stack(states, axis=0),
        "task_descriptions": np.asarray(tasks, dtype=object),
    }


def save_calibration_dataset(
    path: str | Path,
    batches: list[dict],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a version-1 calibration ``.npz`` file."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _batches_to_arrays(batches)
    header: dict[str, Any] = {
        "version": CALIBRATION_FILE_VERSION,
        "num_samples": len(batches),
    }
    if meta:
        header.update(meta)
    np.savez_compressed(path, **arrays, meta_json=json.dumps(header))
    logger.info("Wrote %d calibration samples to %s", len(batches), path)
    return path


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
    """Map harness LIBERO obs → QVLA / pi0.5 processor batch layout."""
    images = obs.get("images", {})
    img_list = list(images.values())
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
    states = obs.get("states", obs.get("state", np.zeros(state_dim, dtype=np.float32)))
    return {
        "images": {"image": image, "wrist_image": wrist_image},
        "states": np.asarray(states, dtype=np.float32),
        "task_description": str(obs.get("task_description", "")),
    }


def allocate_task_quotas(num_samples: int, num_tasks: int = NUM_EXPORT_TASKS) -> list[int]:
    """Split ``num_samples`` across tasks as evenly as possible."""
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive.")
    base, rem = divmod(num_samples, num_tasks)
    return [base + (i < rem) for i in range(num_tasks)]


def allocate_phase_quotas(task_quota: int) -> tuple[int, int, int]:
    """Split one task's quota across early / mid / late (remainder → early, mid, late)."""
    if task_quota < 0:
        raise ValueError("task_quota must be non-negative.")
    base, rem = divmod(task_quota, 3)
    quotas = [base, base, base]
    for i in range(rem):
        quotas[i] += 1
    return quotas[0], quotas[1], quotas[2]


def phase_of_index(t: int, traj_len: int) -> Phase:
    """Map frame index to early/mid/late using full trajectory length thirds."""
    if traj_len <= 0:
        raise ValueError("traj_len must be positive.")
    early_end = traj_len // 3
    mid_end = (2 * traj_len) // 3
    if t < early_end:
        return 0
    if t < mid_end:
        return 1
    return 2


def candidate_indices_for_demo(
    traj_len: int,
    *,
    chunk_size: int,
    jitter: int,
    end_margin: int,
    rng: random.Random,
) -> list[tuple[int, Phase]]:
    """One jittered sample per chunk-aligned anchor; skip near-terminal frames."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if jitter < 0:
        raise ValueError("jitter must be non-negative.")
    if end_margin < 0:
        raise ValueError("end_margin must be non-negative.")
    if traj_len < 2:
        return []

    usable = traj_len - end_margin
    if usable <= 0:
        return []

    out: list[tuple[int, Phase]] = []
    for anchor in range(0, usable, chunk_size):
        delta = rng.randint(-jitter, jitter) if jitter > 0 else 0
        t = min(max(anchor + delta, 0), usable - 1)
        out.append((t, phase_of_index(t, traj_len)))
    return out


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
        for t, phase in candidate_indices_for_demo(
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

        # Cache states per (demo_path, demo_name) while rendering picks.
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
    """Export replayed LIBERO demo observations to a version-1 ``.npz`` file."""
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
    "CALIBRATION_FILE_VERSION",
    "NUM_EXPORT_TASKS",
    "save_calibration_dataset",
    "LiberoExportConfig",
    "allocate_task_quotas",
    "allocate_phase_quotas",
    "phase_of_index",
    "candidate_indices_for_demo",
    "collect_libero_batches",
    "export_libero_dataset",
    "observation_to_pi05_batch",
]
