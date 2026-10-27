"""Version-2 calibration ``.npz`` format shared by all exporters.

Camera layout follows whatever keys the source observation carries
(libero / calvin / robocasa / robotwin / simpler / …).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE_VERSION = 2


def _camera_keys_from_batch(batch: dict) -> list[str]:
    images = batch.get("images", {})
    if not isinstance(images, dict) or not images:
        raise ValueError(
            "Each calibration batch must contain a non-empty 'images' dict; "
            f"got keys {sorted(batch)}."
        )
    return [str(key) for key in images.keys()]


def batches_to_arrays(batches: list[dict]) -> dict[str, Any]:
    """Stack calibration batches with whatever camera keys they carry."""
    if not batches:
        raise ValueError("Cannot save an empty calibration dataset.")
    camera_keys = _camera_keys_from_batch(batches[0])
    per_cam: list[list[np.ndarray]] = [[] for _ in camera_keys]
    states: list[np.ndarray] = []
    tasks: list[str] = []
    for batch in batches:
        keys = _camera_keys_from_batch(batch)
        if keys != camera_keys:
            raise ValueError(
                f"Camera key mismatch across batches: expected {camera_keys}, got {keys}."
            )
        images = batch["images"]
        for index, key in enumerate(camera_keys):
            per_cam[index].append(np.asarray(images[key], dtype=np.uint8))
        states.append(np.asarray(batch["states"], dtype=np.float32).reshape(-1))
        tasks.append(str(batch.get("task_description", "")))
    state_lens = {int(state.size) for state in states}
    if len(state_lens) != 1:
        raise ValueError(f"Inconsistent state dims across batches: {sorted(state_lens)}.")
    stacked = [np.stack(frames, axis=0) for frames in per_cam]
    return {
        "camera_keys": np.asarray(camera_keys, dtype=object),
        "camera_images": np.stack(stacked, axis=1),
        "states": np.stack(states, axis=0),
        "task_descriptions": np.asarray(tasks, dtype=object),
    }


def save_calibration_dataset(
    path: str | Path,
    batches: list[dict],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a version-2 calibration ``.npz`` (arbitrary camera layout)."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = batches_to_arrays(batches)
    camera_keys = [str(key) for key in np.asarray(arrays["camera_keys"]).tolist()]
    header: dict[str, Any] = {
        "version": CALIBRATION_FILE_VERSION,
        "num_samples": len(batches),
        "camera_keys": camera_keys,
    }
    if meta:
        header.update(meta)
        header["version"] = CALIBRATION_FILE_VERSION
        header["num_samples"] = len(batches)
        header["camera_keys"] = camera_keys
    np.savez_compressed(path, **arrays, meta_json=json.dumps(header))
    logger.info(
        "Wrote %d calibration samples (%d cams: %s) to %s",
        len(batches),
        len(camera_keys),
        camera_keys,
        path,
    )
    return path


def observation_to_calibration_batch(obs: dict, *, state_dim: int | None = 8) -> dict:
    """Keep the source observation's own camera keys for calibration export."""
    images_in = obs.get("images", {})
    if not isinstance(images_in, dict) or not images_in:
        raise ValueError("Observation is missing a non-empty images dict.")
    images = {str(key): np.asarray(frame, dtype=np.uint8) for key, frame in images_in.items()}
    states = obs.get("states", obs.get("state"))
    if states is None:
        if state_dim is None:
            raise ValueError("Observation has no state and state_dim was not provided.")
        states = np.zeros(int(state_dim), dtype=np.float32)
    return {
        "images": images,
        "states": np.asarray(states, dtype=np.float32).reshape(-1),
        "task_description": str(obs.get("task_description", "")),
    }


__all__ = [
    "CALIBRATION_FILE_VERSION",
    "batches_to_arrays",
    "observation_to_calibration_batch",
    "save_calibration_dataset",
]
