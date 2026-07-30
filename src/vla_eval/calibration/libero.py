"""
Version-1 ``.npz`` calibration dataset format (QVLA compatible).
Sample LIBERO reset observations via :class:`LIBEROBenchmark`.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark

logger = logging.getLogger(__name__)



CALIBRATION_FILE_VERSION = 1


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
    suite: str = "libero_object"
    seed: int = 7
    env_seed: int | None = None
    num_steps_wait: int = 10
    max_episodes_per_task: int = 50
    max_tasks: int | None = None
    state_dim: int = 8


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


def collect_libero_batches(cfg: LiberoExportConfig, num_samples: int) -> list[dict]:
    """Reset LIBERO episodes and collect ``num_samples`` pi0.5-style batches."""
    bench = LIBEROBenchmark(
        suite=cfg.suite,
        seed=cfg.seed,
        env_seed=cfg.env_seed,
        num_steps_wait=cfg.num_steps_wait,
        send_wrist_image=True,
        send_state=True,
    )
    batches: list[dict] = []
    try:
        tasks = bench.get_tasks()
        if cfg.max_tasks is not None:
            tasks = tasks[: cfg.max_tasks]
        if not tasks:
            raise RuntimeError(f"LIBERO suite {cfg.suite!r} returned zero tasks.")

        slots: list[tuple[int, int]] = []
        for task_idx in range(len(tasks)):
            for ep_idx in range(cfg.max_episodes_per_task):
                slots.append((task_idx, ep_idx))
        rng = random.Random(cfg.seed)
        rng.shuffle(slots)

        for task_idx, ep_idx in slots:
            if len(batches) >= num_samples:
                break
            task = dict(tasks[task_idx])
            task["episode_idx"] = ep_idx
            raw_obs = bench.reset(task)
            obs = bench.make_obs(raw_obs, task)
            batches.append(observation_to_pi05_batch(obs, state_dim=cfg.state_dim))
            logger.info("Collected calibration sample %d/%d", len(batches), num_samples)

        if len(batches) < num_samples:
            logger.warning(
                "Requested %d samples but only collected %d (suite=%s, tasks=%d).",
                num_samples,
                len(batches),
                cfg.suite,
                len(tasks),
            )
    finally:
        bench.cleanup()
    return batches


def export_libero_dataset(
    output: str | Path,
    *,
    num_samples: int = 16,
    config: LiberoExportConfig | None = None,
) -> Path:
    """Export LIBERO reset observations to a version-1 ``.npz`` file."""
    cfg = config or LiberoExportConfig()
    batches = collect_libero_batches(cfg, num_samples)
    return save_calibration_dataset(
        output,
        batches,
        meta={
            "source": "libero",
            "exporter": "vla-eval",
            "suite": cfg.suite,
            "seed": cfg.seed,
            "env_seed": cfg.env_seed if cfg.env_seed is not None else cfg.seed,
            "num_steps_wait": cfg.num_steps_wait,
        },
    )


__all__ = [
    "CALIBRATION_FILE_VERSION",
    "save_calibration_dataset",
    "LiberoExportConfig",
    "collect_libero_batches",
    "export_libero_dataset",
    "observation_to_pi05_batch",
]
