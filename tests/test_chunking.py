"""Tests for action chunk buffer and ensemble functions."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.chunking import ActionChunkBuffer, get_ensemble_fn
from vla_eval.model_servers.predict import PredictModelServer


def test_chunk_buffer_basic():
    fn = get_ensemble_fn("newest")
    buf = ActionChunkBuffer(chunk_size=4, ensemble_fn=fn)
    chunk = np.arange(28, dtype=np.float32).reshape(4, 7)
    buf.push_chunk(chunk)
    for i in range(4):
        action = buf.pop()
        np.testing.assert_array_equal(action, chunk[i])
    assert buf.empty


def test_chunk_buffer_ema_ensemble():
    fn = get_ensemble_fn("ema", ema_alpha=0.5)
    buf = ActionChunkBuffer(chunk_size=2, ensemble_fn=fn)
    chunk1 = np.array([[1.0, 0.0], [2.0, 0.0]])
    chunk2 = np.array([[3.0, 0.0], [4.0, 0.0]])
    buf.push_chunk(chunk1)
    _ = buf.pop()  # consume first action
    # Now 1 action remains: [2.0, 0.0]
    buf.push_chunk(chunk2)
    # Overlap of 1: ema(old=[2,0], new=[3,0], alpha=0.5) = [2.5, 0]
    action = buf.pop()
    assert action is not None
    np.testing.assert_array_almost_equal(action, [2.5, 0.0])


def test_callable_ensemble():
    def my_fn(old, new):
        return old * 0.3 + new * 0.7

    fn = get_ensemble_fn(my_fn)
    assert fn is my_fn


class _CountingChunkServer(PredictModelServer):
    """Returns a fixed chunk; counts how often predict() runs."""

    def __init__(self) -> None:
        super().__init__(chunk_size=4, max_batch_size=1)
        self.predict_calls = 0
        self._chunk = np.arange(28, dtype=np.float32).reshape(4, 7)

    def predict(self, obs, ctx):
        self.predict_calls += 1
        return {"actions": self._chunk.copy()}


@pytest.mark.anyio
async def test_language_change_flushes_open_loop_chunk():
    """Mid-episode prompt change must force a new predict() under chunk_size>1."""
    server = _CountingChunkServer()
    ctx = SessionContext(session_id="s", episode_id="e", mode="sync")
    sent: list[np.ndarray] = []

    async def _capture(action: dict) -> None:
        sent.append(np.asarray(action["actions"]))

    ctx._send_action_fn = _capture

    await server.on_episode_start({}, ctx)
    await server.on_observation({"task_description": "open top drawer"}, ctx)
    await server.on_observation({"task_description": "open top drawer"}, ctx)
    assert server.predict_calls == 1
    assert len(sent) == 2

    await server.on_observation({"task_description": "place apple into top drawer"}, ctx)
    assert server.predict_calls == 2
    assert len(sent) == 3
