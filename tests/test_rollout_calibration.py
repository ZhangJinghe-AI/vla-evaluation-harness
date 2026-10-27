"""Tests for FP-rollout calibration sampling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vla_eval.calibration.rollout import RolloutRecorder, export_rollout_calibration


def _obs(step: int, task: str, *, cameras: dict[str, np.ndarray] | None = None) -> dict:
    if cameras is None:
        cameras = {
            "image": np.full((4, 4, 3), step, dtype=np.uint8),
            "wrist": np.full((4, 4, 3), 255 - step, dtype=np.uint8),
        }
    return {
        "images": cameras,
        "states": np.arange(8, dtype=np.float32) + step,
        "task_description": task,
    }


def _write_episode(
    root,
    task: str,
    episode_idx: int,
    *,
    success: bool,
    length: int = 30,
    cameras_fn=None,
) -> None:
    recorder = RolloutRecorder(root)
    for step in range(length):
        if cameras_fn is None:
            obs = _obs(step, task)
        else:
            obs = _obs(step, task, cameras=cameras_fn(step))
        recorder.note(obs)
    recorder.flush({"name": task, "episode_idx": episode_idx}, success=success)
    recorder.finish()


def test_rollout_write_error_does_not_raise(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    recorder = RolloutRecorder(rollout_dir)
    recorder.note(_obs(0, "task_a"))
    recorder.flush({"name": "task_a", "episode_idx": 0}, success=True)
    recorder.finish()
    recorder.note(_obs(1, "task_a"))
    recorder.flush({"name": "task_a", "episode_idx": 0}, success=True)
    recorder.finish()
    assert len(list(rollout_dir.glob("*.npz"))) == 1


def test_rollout_sampling_keeps_native_camera_keys(tmp_path):
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
        keys = [str(k) for k in data["camera_keys"].tolist()]
        assert keys == ["image", "wrist"]
        assert data["camera_images"].shape == (6, 2, 4, 4, 3)
        assert data["states"].shape == (6, 8)
        meta = json.loads(str(data["meta_json"]))
        descriptions = sorted(data["task_descriptions"].tolist())
    assert meta["version"] == 2
    assert meta["source"] == "rollout"
    assert meta["num_samples"] == 6
    assert meta["camera_keys"] == ["image", "wrist"]
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
        assert data["camera_images"].shape[0] == 3


def test_rollout_sampling_preserves_three_cameras(tmp_path):
    rollout_dir = tmp_path / "rollouts"

    def cams(step: int) -> dict[str, np.ndarray]:
        return {
            "head_camera": np.full((4, 4, 3), step, dtype=np.uint8),
            "left_camera": np.full((4, 4, 3), step + 1, dtype=np.uint8),
            "right_camera": np.full((4, 4, 3), step + 2, dtype=np.uint8),
        }

    _write_episode(rollout_dir, "task_a", 0, success=True, cameras_fn=cams)
    _write_episode(rollout_dir, "task_b", 1, success=True, cameras_fn=cams)

    output = tmp_path / "calib.npz"
    export_rollout_calibration(
        rollout_dir,
        output,
        num_samples=6,
        seed=0,
        chunk_size=10,
        jitter=0,
        end_margin=0,
    )
    with np.load(output, allow_pickle=True) as data:
        keys = [str(k) for k in data["camera_keys"].tolist()]
        assert keys == ["head_camera", "left_camera", "right_camera"]
        assert data["camera_images"].shape == (6, 3, 4, 4, 3)


def test_rollout_sampling_preserves_single_camera(tmp_path):
    rollout_dir = tmp_path / "rollouts"

    def cams(step: int) -> dict[str, np.ndarray]:
        return {"primary": np.full((4, 4, 3), step, dtype=np.uint8)}

    _write_episode(rollout_dir, "task_a", 0, success=True, cameras_fn=cams)
    _write_episode(rollout_dir, "task_b", 1, success=True, cameras_fn=cams)

    output = tmp_path / "calib.npz"
    export_rollout_calibration(
        rollout_dir,
        output,
        num_samples=6,
        seed=0,
        chunk_size=10,
        jitter=0,
        end_margin=0,
    )
    with np.load(output, allow_pickle=True) as data:
        keys = [str(k) for k in data["camera_keys"].tolist()]
        assert keys == ["primary"]
        assert data["camera_images"].shape == (6, 1, 4, 4, 3)


def test_mismatched_camera_keys_raise(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    _write_episode(rollout_dir, "task_a", 0, success=True)
    _write_episode(
        rollout_dir,
        "task_b",
        1,
        success=True,
        cameras_fn=lambda step: {"primary": np.full((4, 4, 3), step, dtype=np.uint8)},
    )
    with pytest.raises(ValueError, match="same camera keys"):
        export_rollout_calibration(
            rollout_dir,
            tmp_path / "calib.npz",
            num_samples=2,
            seed=0,
            chunk_size=10,
            jitter=0,
            end_margin=0,
        )


def test_one_sequence_with_two_prompts_splits_quota(tmp_path):
    """A seq_* file that changes language mid-episode is two prompt groups."""
    rollout_dir = tmp_path / "rollouts"
    recorder = RolloutRecorder(rollout_dir)
    for step in range(60):
        prompt = "pull the handle to open the drawer" if step < 30 else "push the handle to close the drawer"
        recorder.note(_obs(step, prompt))
    recorder.flush({"name": "seq_0000", "episode_idx": 0}, success=True)
    recorder.finish()

    output = tmp_path / "calib.npz"
    export_rollout_calibration(
        rollout_dir,
        output,
        num_samples=4,
        seed=0,
        chunk_size=10,
        jitter=0,
        end_margin=0,
    )
    with np.load(output, allow_pickle=True) as data:
        counts = {}
        for text in data["task_descriptions"].tolist():
            counts[str(text)] = counts.get(str(text), 0) + 1
        meta = json.loads(str(data["meta_json"]))
    assert meta["group"] == "task"
    assert meta["num_tasks"] == 1
    assert meta["num_prompts"] == 2
    assert counts == {
        "pull the handle to open the drawer": 2,
        "push the handle to close the drawer": 2,
    }


def test_short_phase_raises_when_no_unused_frames_remain(tmp_path):
    """Repeat is refused once every candidate frame is already used."""
    rollout_dir = tmp_path / "rollouts"
    _write_episode(rollout_dir, "task_a", 0, success=True, length=30)

    with pytest.raises(ValueError, match="Not enough distinct candidate frames"):
        export_rollout_calibration(
            rollout_dir,
            tmp_path / "calib.npz",
            num_samples=6,
            seed=0,
            chunk_size=10,
            jitter=0,
            end_margin=0,
        )


def test_phase_shortfall_is_reallocated_deterministically(tmp_path):
    """A missing late frame moves to another task that still has one."""
    rollout_dir = tmp_path / "rollouts"
    _write_episode(rollout_dir, "short", 0, success=True, length=20)
    _write_episode(rollout_dir, "short", 1, success=False, length=40)
    _write_episode(rollout_dir, "spare", 0, success=True, length=40)
    _write_episode(rollout_dir, "spare", 1, success=True, length=40)

    def export(path):
        export_rollout_calibration(
            rollout_dir,
            path,
            num_samples=4,
            seed=0,
            chunk_size=10,
            jitter=0,
            end_margin=0,
            success_only=True,
        )

    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    export(first)
    export(second)
    with np.load(first, allow_pickle=True) as data:
        counts: dict[str, int] = {}
        for text in data["task_descriptions"].tolist():
            counts[str(text)] = counts.get(str(text), 0) + 1
        meta = json.loads(str(data["meta_json"]))
        images = data["camera_images"].copy()
    with np.load(second, allow_pickle=True) as data:
        repeat = data["camera_images"]
    assert meta["reallocated_slots"] == 1
    assert counts == {"short": 1, "spare": 3}
    assert np.array_equal(images, repeat)


def test_task_quota_is_not_split_flat_across_prompts(tmp_path):
    """A task with many prompts keeps the same budget as a one-prompt task."""
    rollout_dir = tmp_path / "rollouts"
    recorder = RolloutRecorder(rollout_dir)
    for step in range(30):
        recorder.note(_obs(step, "only"))
    recorder.flush({"name": "beta", "episode_idx": 0}, success=True)
    for episode_idx, prompt in enumerate(("p0", "p1", "p2", "p3")):
        for step in range(30):
            recorder.note(_obs(step, prompt))
        recorder.flush({"name": "alpha", "episode_idx": episode_idx}, success=True)
    recorder.finish()

    output = tmp_path / "calib.npz"
    export_rollout_calibration(
        rollout_dir,
        output,
        num_samples=4,
        seed=0,
        chunk_size=10,
        jitter=0,
        end_margin=0,
    )
    with np.load(output, allow_pickle=True) as data:
        counts: dict[str, int] = {}
        for text in data["task_descriptions"].tolist():
            counts[str(text)] = counts.get(str(text), 0) + 1
        meta = json.loads(str(data["meta_json"]))
    assert meta["group"] == "task"
    assert counts["only"] == 2
    alpha = {name: count for name, count in counts.items() if name != "only"}
    assert sum(alpha.values()) == 2
    assert all(count == 1 for count in alpha.values())


def test_shared_prompts_ignore_task_name(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    recorder = RolloutRecorder(rollout_dir)
    for task_name in ("seq_0000", "seq_0001"):
        for step in range(60):
            prompt = "open" if step < 30 else "close"
            recorder.note(_obs(step, prompt))
        recorder.flush({"name": task_name, "episode_idx": 0}, success=True)
    recorder.finish()

    output = tmp_path / "calib.npz"
    export_rollout_calibration(
        rollout_dir,
        output,
        num_samples=4,
        seed=0,
        chunk_size=10,
        jitter=0,
        end_margin=0,
    )
    with np.load(output, allow_pickle=True) as data:
        counts: dict[str, int] = {}
        for text in data["task_descriptions"].tolist():
            counts[str(text)] = counts.get(str(text), 0) + 1
        meta = json.loads(str(data["meta_json"]))
    assert meta["group"] == "prompt"
    assert counts == {"open": 2, "close": 2}
