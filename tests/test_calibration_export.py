"""Tests for calibration dataset export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vla_eval.calibration.libero import CALIBRATION_FILE_VERSION, save_calibration_dataset, observation_to_pi05_batch


def test_observation_to_pi05_batch():
    agent = np.zeros((256, 256, 3), dtype=np.uint8)
    batch = observation_to_pi05_batch(
        {
            "images": {"agentview": agent, "wrist": agent.copy()},
            "states": np.arange(8, dtype=np.float32),
            "task_description": "pick up the mug",
        }
    )
    assert set(batch["images"]) == {"image", "wrist_image"}
    assert batch["task_description"] == "pick up the mug"


def test_save_calibration_npz_roundtrip(tmp_path):
    batches = [
        {
            "images": {
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
            },
            "states": np.zeros(8, dtype=np.float32),
            "task_description": "task",
        }
    ]
    path = tmp_path / "calib.npz"
    save_calibration_dataset(path, batches, meta={"source": "libero", "suite": "libero_object"})
    with np.load(path, allow_pickle=True) as data:
        meta = json.loads(str(data["meta_json"]))
        assert meta["version"] == CALIBRATION_FILE_VERSION
        assert data["images"].shape == (1, 8, 8, 3)
