"""Naive random-frame calibration export from a local LeRobot dataset."""

from __future__ import annotations

import io
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vla_eval.calibration.format import save_calibration_dataset
from vla_eval.calibration.sampling import assign_shuffled_quotas

_IMAGE_DTYPES = frozenset({"image", "video"})


@dataclass(frozen=True)
class LeRobotExportConfig:
    dataset_dir: str | Path
    num_samples: int
    seed: int = 7
    image_keys: tuple[str, ...] | None = None
    state_key: str = "observation.state"


@dataclass(frozen=True)
class FramePick:
    prompt: str
    episode_index: int
    frame_index: int


def select_random_prompt_frames(
    episodes: Any,
    *,
    num_samples: int,
    seed: int,
) -> list[FramePick]:
    """Split the budget by prompt, then sample each prompt from one episode."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in episodes:
        prompts = [str(prompt).strip() for prompt in row["tasks"]]
        if len(prompts) != 1 or not prompts[0]:
            raise ValueError(
                "Naive LeRobot export requires exactly one non-empty task prompt per episode."
            )
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        if stop <= start:
            raise ValueError(
                f"Episode {row['episode_index']} has invalid frame range [{start}, {stop})."
            )
        by_prompt.setdefault(prompts[0], []).append(row)

    if not by_prompt:
        raise ValueError("LeRobot dataset contains no task prompts.")
    if num_samples < len(by_prompt):
        raise ValueError(
            f"num_samples={num_samples} is smaller than the number of prompts "
            f"({len(by_prompt)}); every prompt must receive at least one sample."
        )

    rng = random.Random(seed)
    quotas = assign_shuffled_quotas(num_samples, sorted(by_prompt), rng)
    picks: list[FramePick] = []
    for prompt in sorted(by_prompt):
        quota = quotas[prompt]
        eligible = [
            row
            for row in by_prompt[prompt]
            if int(row["dataset_to_index"]) - int(row["dataset_from_index"]) >= quota
        ]
        if not eligible:
            raise ValueError(
                f"Prompt {prompt!r} has no episode containing at least {quota} frames."
            )
        row = rng.choice(eligible)
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        for frame_index in rng.sample(range(start, stop), quota):
            picks.append(
                FramePick(
                    prompt=prompt,
                    episode_index=int(row["episode_index"]),
                    frame_index=frame_index,
                )
            )
    return picks


def _selection_episode_indices(root: Path) -> set[int] | None:
    path = root / "selection.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = {
        int(item["episode_index"])
        for items in payload.get("tasks", {}).values()
        for item in items
    }
    if not indices:
        raise ValueError(f"{path} does not contain any episode selections.")
    return indices


def _video_path(root: Path, row: dict[str, Any], key: str) -> Path:
    chunk = int(row[f"videos/{key}/chunk_index"])
    file_index = int(row[f"videos/{key}/file_index"])
    return root / f"videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"


def _data_path(root: Path, row: dict[str, Any]) -> Path:
    return (
        root
        / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    )


def _read_episode_rows(root: Path, image_keys: tuple[str, ...], dtypes: dict[str, str]) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot episode metadata found under {root / 'meta/episodes'}.")
    available = set(pq.read_schema(paths[0]).names)
    columns = [
        "episode_index",
        "dataset_from_index",
        "dataset_to_index",
        "tasks",
    ]
    if "data/chunk_index" in available and "data/file_index" in available:
        columns.extend(["data/chunk_index", "data/file_index"])
    for key in image_keys:
        if dtypes[key] != "video":
            continue
        for suffix in ("chunk_index", "file_index", "from_timestamp"):
            column = f"videos/{key}/{suffix}"
            if column not in available:
                raise ValueError(f"Episode metadata is missing required video column {column!r}.")
            columns.append(column)
    return pa.concat_tables([pq.read_table(path, columns=columns) for path in paths]).to_pylist()


def _filter_episodes(
    root: Path,
    episodes: list[dict[str, Any]],
    *,
    image_keys: tuple[str, ...],
    dtypes: dict[str, str],
) -> list[dict[str, Any]]:
    selected = _selection_episode_indices(root)
    if selected is not None:
        episodes = [row for row in episodes if int(row["episode_index"]) in selected]
        found = {int(row["episode_index"]) for row in episodes}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"selection.json episodes are missing from metadata: {missing[:10]}")

    filtered: list[dict[str, Any]] = []
    for row in episodes:
        if "data/chunk_index" in row and "data/file_index" in row:
            if not _data_path(root, row).is_file():
                continue
        if any(dtypes[key] == "video" and not _video_path(root, row, key).is_file() for key in image_keys):
            continue
        filtered.append(row)
    if not filtered:
        raise FileNotFoundError(
            f"No usable LeRobot episodes with local media were found under {root}."
        )
    return filtered


def _resolve_image_keys(
    info: dict[str, Any], configured: tuple[str, ...] | None
) -> tuple[tuple[str, ...], dict[str, str]]:
    features = info.get("features", {})
    if configured is None:
        keys = tuple(
            key
            for key, feature in features.items()
            if key.startswith("observation.images.")
            and feature.get("dtype") in _IMAGE_DTYPES
        )
    else:
        keys = configured
    if not keys:
        raise ValueError("No LeRobot image features were selected.")
    missing = [key for key in keys if key not in features]
    if missing:
        raise ValueError(f"LeRobot image features do not exist: {missing}.")
    dtypes = {key: str(features[key].get("dtype")) for key in keys}
    unsupported = [key for key, dtype in dtypes.items() if dtype not in _IMAGE_DTYPES]
    if unsupported:
        raise ValueError(
            f"Only LeRobot image/video features are supported: {unsupported}."
        )
    return keys, dtypes


def _decode_image(value: Any, *, root: Path, key: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"Image feature {key!r} must be a bytes/path struct.")
    payload = value.get("bytes")
    path = value.get("path")
    if payload is not None:
        source: Any = io.BytesIO(payload)
    elif isinstance(path, str) and path:
        source = root / path
        if not source.is_file():
            raise FileNotFoundError(f"Image feature {key!r} points to missing file {source}.")
    else:
        raise ValueError(f"Image feature {key!r} contains neither bytes nor a path.")
    with Image.open(source) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(array)


def _decode_video_frame(path: Path, timestamp_sec: float) -> np.ndarray:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-ss",
        f"{timestamp_sec:.6f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Failed to decode frame from {path} at t={timestamp_sec:.6f}: {detail}")
    with Image.open(io.BytesIO(completed.stdout)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(array)


def _sample_to_batch(
    sample: dict[str, Any],
    *,
    root: Path,
    prompt: str,
    episode_index: int,
    episode: dict[str, Any],
    image_keys: tuple[str, ...],
    dtypes: dict[str, str],
    state_key: str,
    fps: float,
) -> dict[str, Any]:
    if int(sample["episode_index"]) != episode_index:
        raise ValueError(
            f"Frame episode {sample['episode_index']} does not match expected episode {episode_index}."
        )
    if state_key not in sample:
        raise ValueError(f"LeRobot sample is missing state feature {state_key!r}.")

    images: dict[str, np.ndarray] = {}
    for key in image_keys:
        camera = key.rsplit(".", 1)[-1]
        if camera in images:
            raise ValueError(f"Image feature names collapse to duplicate camera key {camera!r}.")
        if dtypes[key] == "image":
            if key not in sample:
                raise ValueError(f"LeRobot sample is missing image feature {key!r}.")
            images[camera] = _decode_image(sample[key], root=root, key=key)
            continue
        if "frame_index" not in sample:
            raise ValueError("Video LeRobot export requires a frame_index column in the data files.")
        timestamp = float(episode[f"videos/{key}/from_timestamp"]) + float(sample["frame_index"]) / fps
        images[camera] = _decode_video_frame(_video_path(root, episode, key), timestamp)

    state_array = np.asarray(sample[state_key], dtype=np.float32).reshape(-1)
    if not np.isfinite(state_array).all():
        raise ValueError(f"State feature {state_key!r} contains non-finite values.")
    return {
        "images": images,
        "states": state_array,
        "task_description": prompt,
    }


def _group_frames_by_file(root: Path, frame_indices: set[int]) -> dict[Path, set[int]]:
    import pyarrow.parquet as pq

    remaining = set(frame_indices)
    grouped: dict[Path, set[int]] = {}
    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot data files found under {root / 'data'}.")
    for path in paths:
        indices = {int(value) for value in pq.read_table(path, columns=["index"]).column(0).to_pylist()}
        selected = remaining.intersection(indices)
        if selected:
            grouped[path] = selected
            remaining.difference_update(selected)
            if not remaining:
                break
    if remaining:
        raise ValueError(f"Frames {sorted(remaining)} do not exist in the LeRobot data files.")
    return grouped


def collect_lerobot_batches(
    config: LeRobotExportConfig,
) -> tuple[list[dict[str, Any]], list[FramePick]]:
    root = Path(config.dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset directory does not exist: {root}")

    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset metadata does not exist: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError("Direct calibration export requires a LeRobot v3 dataset.")
    image_keys, dtypes = _resolve_image_keys(info, config.image_keys)
    if config.state_key not in info.get("features", {}):
        raise ValueError(f"LeRobot state feature does not exist: {config.state_key!r}.")
    fps = float(info.get("fps", 0))
    if any(dtype == "video" for dtype in dtypes.values()) and fps <= 0:
        raise ValueError("Video LeRobot export requires a positive fps in meta/info.json.")

    episodes = _filter_episodes(
        root,
        _read_episode_rows(root, image_keys, dtypes),
        image_keys=image_keys,
        dtypes=dtypes,
    )
    episode_by_index = {int(row["episode_index"]): row for row in episodes}
    picks = select_random_prompt_frames(
        episodes,
        num_samples=config.num_samples,
        seed=config.seed,
    )
    frames_by_file = _group_frames_by_file(root, {pick.frame_index for pick in picks})

    import pyarrow.parquet as pq

    embedded_keys = tuple(key for key in image_keys if dtypes[key] == "image")
    samples: dict[int, dict[str, Any]] = {}
    columns = [*embedded_keys, config.state_key, "episode_index", "index"]
    if any(dtype == "video" for dtype in dtypes.values()):
        columns.append("frame_index")
    for path, frame_indices in frames_by_file.items():
        table = pq.read_table(path, columns=columns)
        row_by_frame = {
            int(frame_index): row_index
            for row_index, frame_index in enumerate(table.column("index").to_pylist())
            if int(frame_index) in frame_indices
        }
        missing = sorted(frame_indices - row_by_frame.keys())
        if missing:
            raise ValueError(f"Frames {missing} are missing from {path}.")
        for frame_index, row_index in row_by_frame.items():
            samples[frame_index] = table.slice(row_index, 1).to_pylist()[0]

    batches = [
        _sample_to_batch(
            samples[pick.frame_index],
            root=root,
            prompt=pick.prompt,
            episode_index=pick.episode_index,
            episode=episode_by_index[pick.episode_index],
            image_keys=image_keys,
            dtypes=dtypes,
            state_key=config.state_key,
            fps=fps,
        )
        for pick in picks
    ]
    return batches, picks


def export_lerobot_dataset(output: str | Path, *, config: LeRobotExportConfig) -> Path:
    batches, picks = collect_lerobot_batches(config)
    prompts = sorted({pick.prompt for pick in picks})
    root = Path(config.dataset_dir).expanduser().resolve()
    source_path = root / "qaft_dataset_source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else {}
    return save_calibration_dataset(
        output,
        batches,
        meta={
            "source": "lerobot",
            "exporter": "vla-eval",
            "sampling": "even_prompt_quota_random_frames_from_one_episode",
            "repo_id": source.get("repo_id", "local"),
            "dataset_dir": str(root),
            "seed": config.seed,
            "num_prompts": len(prompts),
            "picks": [
                {
                    "prompt": pick.prompt,
                    "episode_index": pick.episode_index,
                    "frame_index": pick.frame_index,
                }
                for pick in picks
            ],
        },
    )


__all__ = [
    "FramePick",
    "LeRobotExportConfig",
    "collect_lerobot_batches",
    "export_lerobot_dataset",
    "select_random_prompt_frames",
]
