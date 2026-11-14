"""Tests for naive calibration export from LeRobot datasets."""

from __future__ import annotations

import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vla_eval.calibration.lerobot import (
    LeRobotExportConfig,
    export_lerobot_dataset,
    select_random_prompt_frames,
)
from vla_eval.cli.export_calibration import _is_lerobot_dataset


def _episodes():
    rows = []
    start = 0
    for prompt in ("prompt b", "prompt a"):
        for episode in range(4):
            rows.append(
                {
                    "episode_index": len(rows),
                    "dataset_from_index": start,
                    "dataset_to_index": start + 5,
                    "tasks": [prompt],
                }
            )
            start += 5
    return rows


def test_lerobot_dataset_is_inferred_from_local_metadata(tmp_path):
    assert not _is_lerobot_dataset(tmp_path)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    assert _is_lerobot_dataset(tmp_path)


def test_select_random_prompt_frames_is_balanced_single_episode_and_deterministic():
    first = select_random_prompt_frames(_episodes(), num_samples=7, seed=7)
    second = select_random_prompt_frames(_episodes(), num_samples=7, seed=7)
    assert first == second
    counts = {
        prompt: [pick.prompt for pick in first].count(prompt)
        for prompt in ("prompt a", "prompt b")
    }
    assert sorted(counts.values()) == [3, 4]
    for prompt in ("prompt a", "prompt b"):
        selected = [pick for pick in first if pick.prompt == prompt]
        assert len({pick.episode_index for pick in selected}) == 1
        assert len({pick.frame_index for pick in selected}) == counts[prompt]
    rows = {row["episode_index"]: row for row in _episodes()}
    assert all(
        rows[pick.episode_index]["dataset_from_index"]
        <= pick.frame_index
        < rows[pick.episode_index]["dataset_to_index"]
        for pick in first
    )


def test_select_random_prompt_frames_requires_one_sample_per_prompt():
    with pytest.raises(ValueError, match="smaller than the number of prompts"):
        select_random_prompt_frames(_episodes(), num_samples=1, seed=7)


def _png_bytes(value: int) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def test_export_lerobot_dataset_roundtrip(tmp_path):
    rows = _episodes()
    for row in rows:
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0

    info = {
        "codebase_version": "v3.0",
        "features": {
            "observation.images.image": {"dtype": "image"},
            "observation.images.image2": {"dtype": "image"},
            "observation.state": {"dtype": "float32"},
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    (tmp_path / "qaft_dataset_source.json").write_text(
        json.dumps({"repo_id": "test/repo"})
    )
    pq.write_table(
        pa.Table.from_pylist(rows),
        tmp_path / "meta/episodes/chunk-000/file-000.parquet",
    )
    data_rows = []
    for row in rows:
        for frame in range(row["dataset_from_index"], row["dataset_to_index"]):
            data_rows.append(
                {
                    "observation.images.image": {"bytes": _png_bytes(frame), "path": None},
                    "observation.images.image2": {"bytes": _png_bytes(0), "path": None},
                    "observation.state": list(np.arange(8, dtype=np.float32)),
                    "episode_index": row["episode_index"],
                    "index": frame,
                }
            )
    pq.write_table(
        pa.Table.from_pylist(data_rows),
        tmp_path / "data/chunk-000/file-000.parquet",
    )

    output = tmp_path / "naive.npz"
    export_lerobot_dataset(
        output,
        config=LeRobotExportConfig(
            dataset_dir=tmp_path,
            num_samples=6,
            seed=7,
        ),
    )

    with np.load(output, allow_pickle=True) as data:
        meta = json.loads(str(data["meta_json"]))
        assert meta["source"] == "lerobot"
        assert meta["sampling"] == "even_prompt_quota_random_frames_from_one_episode"
        assert meta["num_prompts"] == 2
        assert meta["num_samples"] == 6
        assert data["camera_keys"].tolist() == ["image", "image2"]
        assert data["camera_images"].shape == (6, 2, 2, 2, 3)
        assert data["states"].shape == (6, 8)
        assert sorted(data["task_descriptions"].tolist()) == ["prompt a"] * 3 + [
            "prompt b"
        ] * 3
