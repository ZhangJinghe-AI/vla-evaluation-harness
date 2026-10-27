"""Tests for calibration dataset export."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from vla_eval.calibration.format import CALIBRATION_FILE_VERSION, save_calibration_dataset
from vla_eval.calibration.libero import (
    LiberoExportConfig,
    collect_libero_batches,
    observation_to_pi05_batch,
)
from vla_eval.calibration.sampling import (
    allocate_phase_quotas,
    allocate_task_quotas,
    assign_shuffled_quotas,
    calibration_group,
    candidate_indices_for_demo,
    phase_of_index,
)


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
        assert [str(k) for k in data["camera_keys"].tolist()] == ["image", "wrist_image"]
        assert data["camera_images"].shape == (1, 2, 8, 8, 3)


def test_save_calibration_npz_three_cameras(tmp_path):
    batches = [
        {
            "images": {
                "head_camera": np.zeros((4, 4, 3), dtype=np.uint8),
                "left_camera": np.ones((4, 4, 3), dtype=np.uint8),
                "right_camera": np.full((4, 4, 3), 2, dtype=np.uint8),
            },
            "states": np.zeros(14, dtype=np.float32),
            "task_description": "robotwin",
        }
    ]
    path = tmp_path / "calib.npz"
    save_calibration_dataset(path, batches, meta={"source": "rollout"})
    with np.load(path, allow_pickle=True) as data:
        assert [str(k) for k in data["camera_keys"].tolist()] == [
            "head_camera",
            "left_camera",
            "right_camera",
        ]
        assert data["camera_images"].shape == (1, 3, 4, 4, 3)
        assert data["states"].shape == (1, 14)


def test_allocate_task_quotas_even_split():
    assert allocate_task_quotas(120, 10) == [12] * 10
    assert sum(allocate_task_quotas(125, 10)) == 125
    assert allocate_task_quotas(125, 10)[:5] == [13, 13, 13, 13, 13]
    assert allocate_task_quotas(125, 10)[5:] == [12, 12, 12, 12, 12]


def test_shuffled_quotas_do_not_keep_alphabetical_prefix():
    names = ["a", "b", "c", "d"]
    quotas = assign_shuffled_quotas(2, names, random.Random(0))
    assert sum(quotas.values()) == 2
    assert sorted(name for name, quota in quotas.items() if quota) == ["a", "c"]


def test_calibration_group_uses_prompt_only_when_prompts_cross_tasks():
    assert calibration_group({"pick": {"lift the can"}}) == "task"
    assert (
        calibration_group(
            {
                "seq_0000": {"open", "close"},
                "seq_0001": {"open", "close"},
            }
        )
        == "prompt"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        calibration_group({"a": {"shared"}, "b": {"shared", "only_b"}})


def test_allocate_phase_quotas_balanced():
    # Return order is early, mid, late. Remainder fills mid, then late, then early.
    assert allocate_phase_quotas(12) == (4, 4, 4)
    assert allocate_phase_quotas(11) == (3, 4, 4)
    assert allocate_phase_quotas(10) == (3, 4, 3)
    assert allocate_phase_quotas(2) == (0, 1, 1)
    assert allocate_phase_quotas(1) == (0, 1, 0)


def test_phase_of_index_thirds():
    assert phase_of_index(0, 150) == 0
    assert phase_of_index(49, 150) == 0
    assert phase_of_index(50, 150) == 1
    assert phase_of_index(99, 150) == 1
    assert phase_of_index(100, 150) == 2


def test_candidate_indices_chunk_grid_and_jitter_bounds():
    rng = random.Random(0)
    pairs = candidate_indices_for_demo(
        150, chunk_size=50, jitter=2, end_margin=10, rng=rng
    )
    assert pairs
    assert all(0 <= t < 140 for t, _ in pairs)
    # One candidate per anchor 0,50,100 (150 excluded by end_margin).
    assert len(pairs) == 3
    for t, phase in pairs:
        assert phase == phase_of_index(t, 150)


def test_candidate_indices_zero_jitter_is_exact_grid():
    pairs = candidate_indices_for_demo(
        150, chunk_size=50, jitter=0, end_margin=10, rng=random.Random(0)
    )
    assert [t for t, _ in pairs] == [0, 50, 100]


def test_collect_libero_batches_requires_dataset_dir():
    cfg = LiberoExportConfig(dataset_dir="/path/does/not/exist")
    with pytest.raises(FileNotFoundError):
        collect_libero_batches(cfg, num_samples=30)


def test_collect_libero_batches_accepts_arbitrary_positive_sample_count():
    cfg = LiberoExportConfig(dataset_dir="/path/does/not/exist")
    with pytest.raises(FileNotFoundError):
        collect_libero_batches(cfg, num_samples=64)


def test_collect_libero_batches_rejects_nonpositive_sample_count():
    cfg = LiberoExportConfig(dataset_dir="/path/does/not/exist")
    with pytest.raises(ValueError, match="positive"):
        collect_libero_batches(cfg, num_samples=0)
