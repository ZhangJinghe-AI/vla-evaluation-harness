"""Shared trajectory sampling helpers for calibration export.

Used by FP-rollout export and by the legacy LIBERO demo-replay exporter.
"""

from __future__ import annotations

import random
from typing import Literal

Phase = Literal[0, 1, 2]  # early / mid / late


def allocate_task_quotas(num_samples: int, num_tasks: int) -> list[int]:
    """Split ``num_samples`` across tasks as evenly as possible."""
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    base, rem = divmod(num_samples, num_tasks)
    return [base + (i < rem) for i in range(num_tasks)]


def assign_shuffled_quotas(
    num_samples: int,
    names: list[str],
    rng: random.Random,
) -> dict[str, int]:
    """Even quotas after a seeded shuffle.

    Remainder goes to a random subset of ``names``. Units past the sample
    budget get 0. Callers must not refill those units from somewhere else.
    """
    if len(names) != len(set(names)):
        raise ValueError("Quota names must be unique.")
    ordered = list(names)
    rng.shuffle(ordered)
    return {
        name: quota
        for name, quota in zip(ordered, allocate_task_quotas(num_samples, len(ordered)))
    }


def calibration_group(task_prompts: dict[str, set[str]]) -> Literal["task", "prompt"]:
    """Choose the outer quota unit for one rollout directory.

    ``task``: prompts are private to a ``task_name``, so the budget is
    task, then prompt, then phase.
    ``prompt``: the same prompt is reused under many task names (CALVIN
    ``seq_*``), so task names are ignored.
    An even split between those two patterns is rejected.
    """
    if not task_prompts:
        raise ValueError("No tasks to allocate.")
    prompt_tasks: dict[str, int] = {}
    for prompts in task_prompts.values():
        if not prompts:
            raise ValueError("A task has no prompts.")
        for prompt in prompts:
            if not prompt:
                raise ValueError("A rollout frame has an empty task description.")
            prompt_tasks[prompt] = prompt_tasks.get(prompt, 0) + 1
    shared = sum(count > 1 for count in prompt_tasks.values())
    total = len(prompt_tasks)
    if shared * 2 == total:
        raise ValueError(
            f"Calibration grouping is ambiguous: {shared} of {total} prompts "
            "appear under more than one task_name."
        )
    if shared * 2 > total:
        return "prompt"
    return "task"


def allocate_phase_quotas(task_quota: int) -> tuple[int, int, int]:
    """Split one task's quota across early / mid / late.

    Remainder goes to mid, then late, then early, so a single sample lands
    in the middle third.
    """
    if task_quota < 0:
        raise ValueError("task_quota must be non-negative.")
    base, rem = divmod(task_quota, 3)
    quotas = [base, base, base]
    # 0 early, 1 mid, 2 late
    for phase in (1, 2, 0)[:rem]:
        quotas[phase] += 1
    return quotas[0], quotas[1], quotas[2]


def phase_of_index(t: int, traj_len: int) -> Phase:
    """Map frame index to early/mid/late using full trajectory length thirds."""
    if traj_len <= 0:
        raise ValueError("traj_len must be positive.")
    early_end = traj_len // 3
    mid_end = (2 * traj_len) // 3
    if t < early_end:
        return 0
    if t < mid_end:
        return 1
    return 2


def candidate_indices_for_trajectory(
    traj_len: int,
    *,
    chunk_size: int,
    jitter: int,
    end_margin: int,
    rng: random.Random,
) -> list[tuple[int, Phase]]:
    """One jittered sample per chunk-aligned anchor; skip near-terminal frames."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if jitter < 0:
        raise ValueError("jitter must be non-negative.")
    if end_margin < 0:
        raise ValueError("end_margin must be non-negative.")
    if traj_len < 2:
        return []

    usable = traj_len - end_margin
    if usable <= 0:
        return []

    out: list[tuple[int, Phase]] = []
    for anchor in range(0, usable, chunk_size):
        delta = rng.randint(-jitter, jitter) if jitter > 0 else 0
        t = min(max(anchor + delta, 0), usable - 1)
        out.append((t, phase_of_index(t, traj_len)))
    return out


# Back-compat alias used by older LIBERO demo-export call sites / tests.
candidate_indices_for_demo = candidate_indices_for_trajectory


__all__ = [
    "Phase",
    "allocate_phase_quotas",
    "allocate_task_quotas",
    "assign_shuffled_quotas",
    "calibration_group",
    "candidate_indices_for_demo",
    "candidate_indices_for_trajectory",
    "phase_of_index",
]
