"""SyncEpisodeRunner: waits for inference before stepping."""

from __future__ import annotations

import itertools
import logging
from typing import Any

from vla_eval.benchmarks.base import Benchmark
from vla_eval.runners.base import EpisodeRunner
from vla_eval.types import EpisodeResult, Task

logger = logging.getLogger(__name__)


class SyncEpisodeRunner(EpisodeRunner):
    """Synchronous episode runner: one observation → one action per step.

    Episode flow:
        1. ``benchmark.start_episode(task)``
        2. ``benchmark.get_observation()`` → initial observation.
        3. ``conn.start_episode(task_info)``
        4. Step loop (up to ``max_steps``):
           a. ``conn.act(obs)`` → action from model server
           b. ``benchmark.apply_action(action)``
           c. If ``benchmark.is_done()``: break
           d. ``benchmark.get_observation()`` → next observation
        5. ``conn.end_episode()``
    """

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        rollout_recorder: Any = None,
    ) -> EpisodeResult:
        """Run a synchronous episode."""
        try:
            await benchmark.start_episode(task)
            obs_dict = await benchmark.get_observation()
            self._note_rollout(rollout_recorder, obs_dict)

            # Send only serializable task info to the model server
            task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
            await conn.start_episode({"task": task_info})

            steps = range(max_steps) if max_steps is not None else itertools.count()
            for step in steps:
                action = await conn.act(obs_dict)
                await benchmark.apply_action(action)
                if await benchmark.is_done():
                    break
                obs_dict = await benchmark.get_observation()
                self._note_rollout(rollout_recorder, obs_dict)

            elapsed = await benchmark.get_time()
            metrics = await benchmark.get_result()
            episode_result: dict = {"metrics": metrics, "steps": step + 1, "elapsed_sec": round(elapsed, 3)}

            await conn.end_episode(episode_result)
            self._flush_rollout(
                rollout_recorder,
                task,
                success=bool(episode_result["metrics"].get("success", False)),
            )
            return episode_result
        except Exception:
            if rollout_recorder is not None:
                rollout_recorder.discard()
            raise

    @staticmethod
    def _note_rollout(rollout_recorder: Any, obs_dict: dict) -> None:
        if rollout_recorder is None:
            return
        try:
            rollout_recorder.note(obs_dict)
        except Exception:
            logger.exception("Rollout note failed; evaluation continues")

    @staticmethod
    def _flush_rollout(rollout_recorder: Any, task: Task, *, success: bool) -> None:
        if rollout_recorder is None:
            return
        try:
            rollout_recorder.flush(task, success=success)
        except Exception:
            logger.exception("Rollout flush failed; evaluation result is unchanged")
