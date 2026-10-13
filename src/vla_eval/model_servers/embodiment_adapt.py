"""Observation and action transforms for non-LIBERO embodiments.

Pure NumPy. LIBERO does not call these.
"""

from __future__ import annotations

import numpy as np

from vla_eval.rotation import matrix_to_euler_xyz, quat_to_matrix, quat_wxyz_to_xyzw

# WidowX bridge-frame correction used by the N1.6 SimplerEnv server.
_BRIDGE_DEFAULT_ROT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

# Aloha <-> pi joint conversion. Gripper channels (6, 13) stay positive.
_ROBOTWIN_JOINT_FLIP = np.array(
    [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32
)
_ROBOTWIN_DELTA = np.array(
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], dtype=bool
)


def simpler_widowx_state(eef8: np.ndarray) -> np.ndarray:
    """ManiSkill EE state ``[xyz, quat_wxyz, gripper_open]`` → WidowX 8-D state."""
    state = np.asarray(eef8, dtype=np.float64).reshape(-1)
    if state.size < 8:
        raise ValueError(f"SimplerEnv WidowX state needs 8 dims, got {state.size}.")
    quat_xyzw = quat_wxyz_to_xyzw(state[3:7])
    rpy = matrix_to_euler_xyz(quat_to_matrix(quat_xyzw) @ _BRIDGE_DEFAULT_ROT.T)
    return np.array([*state[:3], *rpy, 0.0, state[7]], dtype=np.float32)


def simpler_google_state(eef8: np.ndarray) -> np.ndarray:
    """ManiSkill EE state → Google Robot ``[xyz, quat_xyzw, gripper_closed]``."""
    state = np.asarray(eef8, dtype=np.float64).reshape(-1)
    if state.size < 8:
        raise ValueError(f"SimplerEnv Google state needs 8 dims, got {state.size}.")
    quat_xyzw = quat_wxyz_to_xyzw(state[3:7])
    gripper_closed = 1.0 - float(state[7])
    return np.array([*state[:3], *quat_xyzw, gripper_closed], dtype=np.float32)


def _unnormalize(value: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Map a unit-interval value back into ``[min_val, max_val]``.

    This is the inverse of :func:`_normalize`. OpenPI's Aloha gripper
    conversion calls it on a ``[0, 1]`` gripper before the linear-to-angular
    map. Using the normalize formula here saturates every open gripper to the
    same angle, so the policy never sees the gripper close.
    """
    return value * (max_val - min_val) + min_val


def _normalize(value: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return (value - min_val) / (max_val - min_val)


def _gripper_to_angular(value: np.ndarray) -> np.ndarray:
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)
    cosine = (0.022**2 + value**2 - 0.036**2) / (2 * 0.022 * value)
    radians = np.arcsin(np.clip(cosine, -1.0, 1.0))
    return _normalize(radians, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value: np.ndarray) -> np.ndarray:
    value = value + 0.5476
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def robotwin_state_to_pi(state: np.ndarray) -> np.ndarray:
    """RoboTwin 14-D qpos → pi joint space (flip + angular gripper)."""
    joints = np.asarray(state, dtype=np.float32).reshape(-1)
    if joints.size != 14:
        raise ValueError(f"RoboTwin state must be 14-D, got {joints.size}.")
    out = _ROBOTWIN_JOINT_FLIP * joints
    out[[6, 13]] = _gripper_to_angular(out[[6, 13]])
    return out.astype(np.float32)


def robotwin_pi_actions_to_env(actions: np.ndarray, pi_state: np.ndarray) -> np.ndarray:
    """pi-space delta joints → RoboTwin absolute qpos."""
    out = np.asarray(actions, dtype=np.float32).copy()
    if out.shape[-1] != 14:
        raise ValueError(f"RoboTwin actions must end in 14 dims, got {out.shape}.")
    state = np.asarray(pi_state, dtype=np.float32).reshape(-1)
    out[..., _ROBOTWIN_DELTA] += state[_ROBOTWIN_DELTA]
    out *= _ROBOTWIN_JOINT_FLIP
    out[..., [6, 13]] = _gripper_from_angular(out[..., [6, 13]])
    return out
