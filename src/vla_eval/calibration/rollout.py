"""Save FP-policy rollouts during eval, then sample a version-1 calibration npz.

Recording is off unless the eval config sets ``rollout_dir``. Sampling reuses
the LIBERO exporter's chunk grid, temporal jitter, and early/mid/late quotas.
"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.calibration.libero import (
    allocate_phase_quotas,
    allocate_task_quotas,
    candidate_indices_for_demo,
    observation_to_pi05_batch,
    save_calibration_dataset,
)

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^\w.\-]+")


def _safe_name(text: str) -> str:
    cleaned = _SAFE_NAME.sub("_", text).strip("._")
    return cleaned[:80] or "task"


def _state_vector(obs: dict[str, Any]) -> np.ndarray:
    for key in ("states", "state", "joint_state"):
        value = obs.get(key)
        if value is None:
            continue
        return np.asarray(value, dtype=np.float32).reshape(-1)
    return np.zeros(0, dtype=np.float32)


class RolloutRecorder:
    """Buffer one episode of model-facing observations, then write ``.npz``."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list[dict[str, Any]] = []

    def note(self, obs: dict[str, Any]) -> None:
        images = obs.get("images") or {}
        if not isinstance(images, dict) or not images:
            raise ValueError("Rollout observation is missing images.")
        keys = list(images.keys())
        cams = [np.asarray(images[key], dtype=np.uint8) for key in keys]
        self._frames.append(
            {
                "keys": keys,
                "cams": cams,
                "state": _state_vector(obs),
                "task_description": str(obs.get("task_description", "")),
            }
        )

    def discard(self) -> None:
        self._frames.clear()

    def flush(self, task: dict[str, Any], *, success: bool) -> Path | None:
        frames = self._frames
        self._frames = []
        if not frames:
            return None

        keys = list(frames[0]["keys"])
        if any(list(frame["keys"]) != keys for frame in frames):
            raise ValueError("Camera keys changed within one rollout episode.")
        stacked = [np.stack([frame["cams"][i] for frame in frames], axis=0) for i in range(len(keys))]
        images = np.stack(stacked, axis=1)
        states = np.stack([frame["state"] for frame in frames], axis=0)
        descriptions = np.asarray([frame["task_description"] for frame in frames], dtype=object)

        task_name = str(task.get("name") or descriptions[0] or "task")
        episode_idx = int(task.get("episode_idx", 0))
        status = "success" if success else "fail"
        stem = f"{_safe_name(task_name)}_ep{episode_idx:04d}_{status}"
        final_path = self.output_dir / f"{stem}.npz"
        if final_path.exists():
            raise FileExistsError(f"Rollout already exists: {final_path}")

        tmp_path = self.output_dir / f".rollout-{uuid.uuid4().hex[:12]}.npz"
        np.savez_compressed(
            tmp_path,
            camera_keys=np.asarray(keys, dtype=object),
            images=images,
            states=states,
            task_descriptions=descriptions,
            task_name=np.asarray(task_name),
            success=np.asarray(success),
        )
        tmp_path.replace(final_path)
        meta = {
            "file": final_path.name,
            "task_name": task_name,
            "length": int(images.shape[0]),
            "success": bool(success),
        }
        final_path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        logger.info("Saved rollout: %s (%d frames)", final_path.name, images.shape[0])
        return final_path


def _load_meta(path: Path) -> dict[str, Any]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    npz_path = path.parent / str(meta["file"])
    if npz_path.suffix != ".npz" or not npz_path.is_file():
        raise FileNotFoundError(f"Missing rollout next to {path}")
    meta["npz_path"] = npz_path
    return meta


def _frame_batch(npz_path: Path, index: int, *, state_dim: int) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        keys = [str(key) for key in data["camera_keys"].tolist()]
        frame = data["images"][index]
        images = {key: frame[i] for i, key in enumerate(keys)}
        state = np.asarray(data["states"][index], dtype=np.float32).reshape(-1)
        if state.size == 0:
            state = np.zeros(state_dim, dtype=np.float32)
        description = str(data["task_descriptions"][index])
    return observation_to_pi05_batch(
        {"images": images, "states": state, "task_description": description},
        state_dim=state_dim,
    )


def export_rollout_calibration(
    rollout_dir: str | Path,
    output: str | Path,
    *,
    num_samples: int = 120,
    seed: int = 7,
    chunk_size: int = 50,
    jitter: int = 2,
    end_margin: int = 10,
    success_only: bool = False,
    state_dim: int = 8,
) -> Path:
    """Sample saved rollouts into a version-1 calibration ``.npz``."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    root = Path(rollout_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Rollout directory does not exist: {root}")

    episodes = [_load_meta(path) for path in sorted(root.glob("*.meta.json"))]
    if success_only:
        episodes = [ep for ep in episodes if ep["success"]]
    if not episodes:
        raise ValueError(f"No rollout episodes found in {root}")

    by_task: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_task.setdefault(str(episode["task_name"]), []).append(episode)
    task_names = sorted(by_task)
    quotas = allocate_task_quotas(num_samples, len(task_names))
    rng = random.Random(seed)
    picks: list[tuple[Path, int]] = []

    for task_name, task_quota in zip(task_names, quotas):
        if task_quota == 0:
            continue
        phase_quotas = allocate_phase_quotas(task_quota)
        pools: list[list[tuple[Path, int]]] = [[], [], []]
        order = list(by_task[task_name])
        rng.shuffle(order)
        for episode in order:
            if all(len(pools[phase]) >= phase_quotas[phase] for phase in range(3)):
                break
            for index, phase in candidate_indices_for_demo(
                int(episode["length"]),
                chunk_size=chunk_size,
                jitter=jitter,
                end_margin=end_margin,
                rng=rng,
            ):
                pools[phase].append((episode["npz_path"], index))
        for phase, need in enumerate(phase_quotas):
            if len(pools[phase]) < need:
                raise ValueError(
                    f"Not enough phase-{phase} frames for task {task_name!r}: "
                    f"need {need}, have {len(pools[phase])}. "
                    "Run more episodes, or lower num_samples / chunk_size."
                )
            picks.extend(rng.sample(pools[phase], need))

    batches = [_frame_batch(path, index, state_dim=state_dim) for path, index in picks]
    if len(batches) != num_samples:
        raise RuntimeError(f"Expected {num_samples} calibration samples, got {len(batches)}.")
    return save_calibration_dataset(
        output,
        batches,
        meta={
            "source": "rollout",
            "exporter": "vla-eval",
            "sampling": "task_phase_balanced_chunk_grid_jitter",
            "rollout_dir": str(root),
            "seed": seed,
            "chunk_size": chunk_size,
            "jitter": jitter,
            "end_margin": end_margin,
            "success_only": success_only,
            "num_tasks": len(task_names),
        },
    )
