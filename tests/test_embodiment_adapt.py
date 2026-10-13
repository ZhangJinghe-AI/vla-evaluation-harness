"""Pure transforms for SimplerEnv, RoboTwin, and RoboCasa. No simulator."""

from __future__ import annotations

import numpy as np

from vla_eval.benchmarks.robocasa.benchmark import _call_create_env, lerobot12_to_env_action
from vla_eval.model_servers.embodiment_adapt import (
    robotwin_pi_actions_to_env,
    robotwin_state_to_pi,
    simpler_google_state,
    simpler_widowx_state,
)


def test_simpler_states_are_8d():
    eef = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4], dtype=np.float32)
    widowx = simpler_widowx_state(eef)
    google = simpler_google_state(eef)
    assert widowx.shape == (8,)
    assert google.shape == (8,)
    assert widowx[6] == 0.0
    assert np.isclose(google[7], 0.6)
    np.testing.assert_allclose(google[3:7], [0.0, 0.0, 0.0, 1.0])


def test_robotwin_round_trip_shapes():
    state = np.linspace(0.02, 0.05, 14, dtype=np.float32)
    pi_state = robotwin_state_to_pi(state)
    actions = np.zeros((4, 14), dtype=np.float32)
    env_actions = robotwin_pi_actions_to_env(actions, pi_state)
    assert pi_state.shape == (14,)
    assert env_actions.shape == (4, 14)


def test_robotwin_gripper_matches_openpi_angular_range():
    """Open gripper (1) and closed gripper (0) must land on the training q01/q99."""
    closed = robotwin_state_to_pi(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    opened = robotwin_state_to_pi(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(closed[[6, 13]], -1.0799, atol=1e-3)
    np.testing.assert_allclose(opened[[6, 13]], 0.9457, atol=1e-3)


def test_lerobot12_action_keeps_controller_order():
    flat = np.arange(12, dtype=np.float64)
    np.testing.assert_allclose(lerobot12_to_env_action(flat, 12), flat)


def test_create_env_drops_unknown_split():
    def create_env(env_name, seed=None):
        return (env_name, seed)

    assert _call_create_env(create_env, env_name="Task", split="pretrain", seed=0) == ("Task", 0)


def test_create_env_maps_pretrain_split_to_legacy_a():
    def create_env(env_name, obj_instance_split=None):
        return obj_instance_split

    assert _call_create_env(create_env, env_name="Task", split="pretrain") == "A"


def test_robocasa_episode_seed_is_stable_under_sharding_order():
    """Same (task_id, episode_idx) must map to one seed regardless of call order."""
    from vla_eval.benchmarks.robocasa.benchmark import RoboCasaBenchmark

    bench = RoboCasaBenchmark(tasks=["OpenSingleDoor"], seed=7)
    first = bench._episode_seed({"task_id": 2, "episode_idx": 5})
    second = bench._episode_seed({"task_id": 2, "episode_idx": 5})
    other = bench._episode_seed({"task_id": 2, "episode_idx": 6})
    assert first == second == 7 + 100_000 * 2 + 5
    assert other != first


def test_simpler_enables_physx_enhanced_determinism(monkeypatch):
    from types import SimpleNamespace

    import vla_eval.benchmarks.simpler.benchmark as simpler_mod

    class _FakeSceneConfig:
        enable_enhanced_determinism = False

    class _FakeBaseEnv:
        def _get_default_scene_config(self):
            return _FakeSceneConfig()

    fake_mod = SimpleNamespace(BaseEnv=_FakeBaseEnv)
    monkeypatch.setitem(
        __import__("sys").modules,
        "mani_skill2_real2sim.envs.sapien_env",
        fake_mod,
    )
    # Force re-patch on a clean class each call.
    if hasattr(_FakeBaseEnv, "_vla_enhanced_determinism"):
        delattr(_FakeBaseEnv, "_vla_enhanced_determinism")
    simpler_mod._enable_physx_enhanced_determinism()
    cfg = _FakeBaseEnv()._get_default_scene_config()
    assert cfg.enable_enhanced_determinism is True
    # Idempotent.
    simpler_mod._enable_physx_enhanced_determinism()
    assert _FakeBaseEnv._vla_enhanced_determinism is True
