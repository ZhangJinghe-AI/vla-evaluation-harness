"""Tests for FP-rollout calibration sampling."""

from __future__ import annotations

import json

import numpy as np

from vla_eval.calibration.rollout import RolloutRecorder, export_rollout_calibration


def _obs(step: int, task: str) -> dict:
    image = np.full((4, 4, 3), step, dtype=np.uint8)
    wrist = np.full((4, 4, 3), 255 - step, dtype=np.uint8)
    return {
        "images": {"image": image, "wrist": wrist},
        "states": np.arange(8, dtype=np.float32) + step,
        "task_description": task,
    }


def _write_episode(root, task: str, episode_idx: int, *, success: bool, length: int = 30) -> None:
    recorder = RolloutRecorder(root)
    for step in range(length):
        recorder.note(_obs(step, task))
    recorder.flush({"name": task, "episode_idx": episode_idx}, success=success)


def test_rollout_sampling_matches_chunk_grid(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    _write_episode(rollout_dir, "task_a", 0, success=True)
    _write_episode(rollout_dir, "task_b", 1, success=False)

    output = tmp_path / "calib.npz"
    export_rollout_calibration(
        rollout_dir,
        output,
        num_samples=6,
        seed=0,
        chunk_size=10,
        jitter=0,
        end_margin=0,
        state_dim=8,
    )
    with np.load(output, allow_pickle=True) as data:
        assert data["images"].shape == (6, 4, 4, 3)
        assert data["wrist_images"].shape == (6, 4, 4, 3)
        assert data["states"].shape == (6, 8)
        meta = json.loads(str(data["meta_json"]))
        descriptions = sorted(data["task_descriptions"].tolist())
    assert meta["source"] == "rollout"
    assert meta["num_samples"] == 6
    assert descriptions == ["task_a", "task_a", "task_a", "task_b", "task_b", "task_b"]


def test_success_only_drops_failed_episodes(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    _write_episode(rollout_dir, "task_a", 0, success=True)
    _write_episode(rollout_dir, "task_a", 1, success=False)

    output = tmp_path / "calib.npz"
    export_rollout_calibration(
        rollout_dir,
        output,
        num_samples=3,
        seed=1,
        chunk_size=10,
        jitter=0,
        end_margin=0,
    )
    with np.load(output, allow_pickle=True) as data:
        assert data["images"].shape[0] == 3
