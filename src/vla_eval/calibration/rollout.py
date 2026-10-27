"""FP-policy rollouts → calibration ``.npz`` for any benchmark.

Recording is off unless the eval config sets ``rollout_dir``. Sampling reuses
the shared chunk grid / temporal jitter / early-mid-late quotas. Camera keys
are kept as recorded by each benchmark.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.calibration.format import (
    observation_to_calibration_batch,
    save_calibration_dataset,
)
from vla_eval.calibration.sampling import (
    allocate_phase_quotas,
    assign_shuffled_quotas,
    calibration_group,
    candidate_indices_for_trajectory,
)

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^\w.\-]+")
_PHASE_NAMES = ("early", "mid", "late")


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
    """Buffer one episode of model-facing observations, then write ``.npz``.

    Recording is a side channel. ``note`` copies arrays so the eval loop keeps
    its own buffers, and ``flush`` writes on a background thread. Disk errors
    are logged and do not propagate to the episode result.
    """

    def __init__(self, output_dir: str | Path, *, shard_id: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_id = shard_id
        self._frames: list[dict[str, Any]] = []
        self._skip_episode = False
        self._writers: list[threading.Thread] = []

    def note(self, obs: dict[str, Any]) -> None:
        if self._skip_episode:
            return
        try:
            images = obs.get("images") or {}
            if not isinstance(images, dict) or not images:
                raise ValueError("Rollout observation is missing images.")
            keys = list(images.keys())
            cams = [np.array(images[key], dtype=np.uint8, copy=True) for key in keys]
            self._frames.append(
                {
                    "keys": keys,
                    "cams": cams,
                    "state": _state_vector(obs).copy(),
                    "task_description": str(obs.get("task_description", "")),
                }
            )
        except Exception:
            logger.exception("Failed to buffer a rollout frame; this episode will not be saved")
            self._frames.clear()
            self._skip_episode = True

    def discard(self) -> None:
        self._frames.clear()
        self._skip_episode = False

    def finish(self) -> None:
        """Wait until background writes from this recorder have finished."""
        for writer in self._writers:
            writer.join()
        self._writers.clear()

    def flush(self, task: dict[str, Any], *, success: bool) -> None:
        frames = self._frames
        self._frames = []
        skip = self._skip_episode
        self._skip_episode = False
        if skip or not frames:
            return
        writer = threading.Thread(
            target=self._write_episode,
            args=(frames, dict(task), bool(success)),
            daemon=True,
        )
        self._writers.append(writer)
        writer.start()

    def _write_episode(self, frames: list[dict[str, Any]], task: dict[str, Any], success: bool) -> None:
        try:
            self._write_episode_file(frames, task, success)
        except Exception:
            logger.exception("Failed to save rollout; evaluation result is unchanged")

    def _write_episode_file(self, frames: list[dict[str, Any]], task: dict[str, Any], success: bool) -> Path:
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
        shard = f"_shard{self.shard_id}" if self.shard_id is not None else ""
        stem = f"{_safe_name(task_name)}_ep{episode_idx:04d}{shard}_{status}"
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


def _episode_camera_keys(npz_path: Path) -> list[str]:
    with np.load(npz_path, allow_pickle=True) as data:
        return [str(key) for key in data["camera_keys"].tolist()]


def _episode_descriptions(npz_path: Path) -> list[str]:
    with np.load(npz_path, allow_pickle=True) as data:
        return [str(text) for text in data["task_descriptions"].tolist()]


def _prompt_segments(descriptions: list[str]) -> list[tuple[str, int, int]]:
    """Contiguous ``(prompt, start, end)`` spans. ``end`` is exclusive."""
    if not descriptions:
        return []
    segments: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(descriptions) + 1):
        if index == len(descriptions) or descriptions[index] != descriptions[start]:
            segments.append((descriptions[start], start, index))
            start = index
    return segments


def _frame_batch(npz_path: Path, index: int, *, state_dim: int) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        keys = [str(key) for key in data["camera_keys"].tolist()]
        frame = data["images"][index]
        images = {key: frame[i] for i, key in enumerate(keys)}
        state = np.asarray(data["states"][index], dtype=np.float32).reshape(-1)
        if state.size == 0:
            state = np.zeros(state_dim, dtype=np.float32)
        description = str(data["task_descriptions"][index])
    return observation_to_calibration_batch(
        {"images": images, "states": state, "task_description": description},
        state_dim=state_dim,
    )


def _frames_for_prompt(
    pools: dict[str, dict[str, list[dict[tuple[Path, int], None]]]],
    prompt: str,
    task_names: list[str],
) -> list[list[tuple[Path, int]]]:
    phases: list[list[tuple[Path, int]]] = [[], [], []]
    seen: list[set[tuple[Path, int]]] = [set(), set(), set()]
    for task_name in task_names:
        task_pool = pools.get(task_name, {}).get(prompt)
        if task_pool is None:
            raise ValueError(f"Prompt {prompt!r} has no frames under task {task_name!r}.")
        for phase in range(3):
            for frame in task_pool[phase]:
                if frame in seen[phase]:
                    continue
                seen[phase].add(frame)
                phases[phase].append(frame)
    return phases


def _reallocate_shortfalls(
    shortfalls: list[int],
    units: list[tuple[str, list[list[tuple[Path, int]]]]],
    used: set[tuple[Path, int]],
    picks: list[tuple[Path, int]],
    rng: random.Random,
) -> int:
    """Place unmet phase slots onto unused candidate frames.

    The same phase is tried first. Remaining phases follow mid, late, early.
    The recipient unit and frame are chosen with ``rng``, so one seed repeats.
    """
    placed = 0
    for phase in shortfalls:
        phase_order = [phase] + [item for item in (1, 2, 0) if item != phase]
        found = False
        for try_phase in phase_order:
            options: list[tuple[str, list[tuple[Path, int]]]] = []
            for label, frames in units:
                available = [frame for frame in frames[try_phase] if frame not in used]
                if available:
                    options.append((label, available))
            if not options:
                continue
            options.sort(key=lambda item: item[0])
            _, available = rng.choice(options)
            frame = rng.choice(available)
            picks.append(frame)
            used.add(frame)
            placed += 1
            found = True
            break
        if not found:
            break
    return placed


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
    """Sample saved rollouts into a version-2 calibration ``.npz``.

    One directory is one calibration set. When prompts stay inside a single
    ``task_name``, the budget is split task → prompt → phase. When the same
    prompt is reused across task names, task names are ignored and the budget
    is split prompt → phase.     Extra units are dropped by a seeded shuffle. A phase shortfall is filled
    from unused candidate frames, preferring the same phase on another unit.
    Frames are never repeated. Export raises only when those candidates still
    cannot reach ``num_samples``. The same seed always selects the same frames.
    Early/mid/late is measured inside each contiguous prompt span.
    """
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

    camera_keys = _episode_camera_keys(episodes[0]["npz_path"])
    for episode in episodes[1:]:
        keys = _episode_camera_keys(episode["npz_path"])
        if keys != camera_keys:
            raise ValueError(
                "All rollout episodes must share the same camera keys; "
                f"got {camera_keys} vs {keys} ({episode['npz_path'].name})."
            )

    # task -> prompt -> phase -> distinct (path, index)
    pools: dict[str, dict[str, list[dict[tuple[Path, int], None]]]] = {}
    task_prompts: dict[str, set[str]] = {}
    jitter_rng = random.Random(seed)
    alloc_rng = random.Random(seed)
    for episode in episodes:
        task_name = str(episode.get("task_name") or "")
        if not task_name:
            raise ValueError(f"Rollout meta is missing task_name: {episode['npz_path'].name}")
        descriptions = _episode_descriptions(episode["npz_path"])
        for prompt, start, end in _prompt_segments(descriptions):
            task_prompts.setdefault(task_name, set()).add(prompt)
            phase_pools = pools.setdefault(task_name, {}).setdefault(prompt, [{}, {}, {}])
            for local_index, phase in candidate_indices_for_trajectory(
                end - start,
                chunk_size=chunk_size,
                jitter=jitter,
                end_margin=end_margin,
                rng=jitter_rng,
            ):
                phase_pools[int(phase)][(episode["npz_path"], start + local_index)] = None

    group = calibration_group(task_prompts)
    prompt_names = {prompt for prompts in task_prompts.values() for prompt in prompts}
    if group == "prompt":
        quotas = assign_shuffled_quotas(num_samples, sorted(prompt_names), alloc_rng)
        selections = []
        for prompt, quota in quotas.items():
            if quota == 0:
                continue
            owners = [task for task, prompts in task_prompts.items() if prompt in prompts]
            selections.append((prompt, quota, owners))
    else:
        task_quotas = assign_shuffled_quotas(num_samples, sorted(task_prompts), alloc_rng)
        selections = []
        for task_name, task_quota in task_quotas.items():
            if task_quota == 0:
                continue
            prompt_quotas = assign_shuffled_quotas(
                task_quota, sorted(task_prompts[task_name]), alloc_rng
            )
            for prompt, quota in prompt_quotas.items():
                if quota > 0:
                    selections.append((prompt, quota, [task_name]))

    if group == "prompt":
        units = [
            (
                prompt,
                _frames_for_prompt(
                    pools,
                    prompt,
                    [task for task, prompts in task_prompts.items() if prompt in prompts],
                ),
            )
            for prompt in sorted(prompt_names)
        ]
    else:
        units = [
            (f"{task_name}\0{prompt}", _frames_for_prompt(pools, prompt, [task_name]))
            for task_name in sorted(task_prompts)
            for prompt in sorted(task_prompts[task_name])
        ]
    picks: list[tuple[Path, int]] = []
    used: set[tuple[Path, int]] = set()
    shortfalls: list[int] = []
    for prompt, quota, task_names in selections:
        phase_quotas = allocate_phase_quotas(quota)
        merged = _frames_for_prompt(pools, prompt, task_names)
        for phase, need in enumerate(phase_quotas):
            if need == 0:
                continue
            available = [frame for frame in merged[phase] if frame not in used]
            take = min(need, len(available))
            if take:
                chosen = alloc_rng.sample(available, take)
                picks.extend(chosen)
                used.update(chosen)
            shortfalls.extend([phase] * (need - take))
    reallocated = _reallocate_shortfalls(shortfalls, units, used, picks, alloc_rng)
    if len(picks) != num_samples:
        raise ValueError(
            f"Not enough distinct candidate frames to reach {num_samples}: "
            f"filled {len(picks)} "
            f"(chunk_size={chunk_size}, end_margin={end_margin}, success_only={success_only})."
        )

    batches = [_frame_batch(path, index, state_dim=state_dim) for path, index in picks]
    if len(batches) != num_samples:
        raise RuntimeError(f"Expected {num_samples} calibration samples, got {len(batches)}.")
    return save_calibration_dataset(
        output,
        batches,
        meta={
            "source": "rollout",
            "exporter": "vla-eval",
            "sampling": f"{group}_then_phase_chunk_grid_jitter",
            "group": group,
            "rollout_dir": str(root),
            "seed": seed,
            "chunk_size": chunk_size,
            "jitter": jitter,
            "end_margin": end_margin,
            "success_only": success_only,
            "num_tasks": len(task_prompts),
            "num_prompts": len(prompt_names),
            "reallocated_slots": reallocated,
            "camera_keys": camera_keys,
        },
    )


__all__ = [
    "RolloutRecorder",
    "export_rollout_calibration",
]
