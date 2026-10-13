"""Build a LeRobot-style pi0.5 folder from an OpenPI safetensors checkpoint.

The phyai engine reads ``config.json`` plus ``policy_*processor.json``.
OpenPI releases ship ``model.safetensors`` and a separate ``norm_stats.json``.
This writes a cache directory next to the checkpoint and symlinks the weights.
An existing LeRobot checkpoint (already has both json files) is returned as-is.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_TOKENIZER = "google/paligemma-3b-pt-224"


def prepare_pi05_checkpoint(
    source: str | Path,
    *,
    norm_stats: str | Path | None = None,
    chunk_size: int | None = None,
    action_dim: int | None = None,
    state_dim: int | None = None,
    empty_cameras: int | None = None,
    num_inference_steps: int = 10,
    n_action_steps: int | None = None,
    discrete_state_input: bool = True,
) -> Path:
    source = Path(source).expanduser().resolve()
    if (source / "config.json").is_file() and (source / "policy_preprocessor.json").is_file():
        return source
    if not (source / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"{source} is not a LeRobot pi0.5 checkpoint (missing config.json) "
            "and has no model.safetensors to adapt from OpenPI."
        )
    if norm_stats is None or chunk_size is None or action_dim is None or state_dim is None:
        raise ValueError(
            f"{source} has no LeRobot processor files. Set openpi_norm_stats, "
            "openpi_chunk_size, openpi_action_dim, and openpi_state_dim."
        )
    if empty_cameras is None:
        raise ValueError("openpi_empty_cameras is required for an OpenPI pi0.5 checkpoint.")

    cache = source / ".phyai_lerobot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    link = cache / "model.safetensors"
    target = source / "model.safetensors"
    if link.is_symlink() or link.exists():
        if not link.is_symlink() or link.resolve() != target.resolve():
            link.unlink()
            link.symlink_to(target)
    else:
        link.symlink_to(target)

    stats = _read_openpi_stats(Path(norm_stats))
    _write_stats(
        cache,
        stats,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_state_input=discrete_state_input,
    )
    (cache / "config.json").write_text(
        json.dumps(_config(
            chunk_size=chunk_size,
            action_dim=action_dim,
            state_dim=state_dim,
            empty_cameras=empty_cameras,
            num_inference_steps=num_inference_steps,
            n_action_steps=n_action_steps,
        )),
        encoding="utf-8",
    )
    return cache


def _read_openpi_stats(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    node = payload.get("norm_stats", payload)
    if "state" not in node or "actions" not in node:
        raise ValueError(f"{path} is missing state/actions norm stats.")
    return {"state": node["state"], "actions": node["actions"]}


def _write_stats(
    cache: Path,
    stats: dict[str, dict[str, list[float]]],
    *,
    state_dim: int,
    action_dim: int,
    discrete_state_input: bool = True,
) -> None:
    from safetensors.numpy import save_file

    state = {key: np.asarray(stats["state"][key], dtype=np.float32) for key in ("q01", "q99", "mean", "std")}
    action = {key: np.asarray(stats["actions"][key], dtype=np.float32) for key in ("q01", "q99", "mean", "std")}
    if state["mean"].shape != (state_dim,) or action["mean"].shape != (action_dim,):
        raise ValueError(
            f"norm stats dims state={state['mean'].shape} action={action['mean'].shape} "
            f"do not match state_dim={state_dim} action_dim={action_dim}."
        )
    pre_name = "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_name = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    pre_tensors = {f"observation.state.{key}": value for key, value in state.items()}
    pre_tensors.update({f"action.{key}": value for key, value in action.items()})
    save_file(pre_tensors, cache / pre_name)
    save_file({f"action.{key}": value for key, value in action.items()}, cache / post_name)
    norm_map = {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}
    (cache / "policy_preprocessor.json").write_text(json.dumps({
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": 1e-8,
                    "features": {
                        "observation.state": {"type": "STATE", "shape": [state_dim]},
                        "action": {"type": "ACTION", "shape": [action_dim]},
                    },
                    "norm_map": norm_map,
                },
                "state_file": pre_name,
            },
            {
                "registry_name": "pi05_prepare_state_tokenizer_processor_step",
                "config": {}
                if discrete_state_input
                else {"discrete_state_input": False},
            },
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "max_length": 200,
                    "task_key": "task",
                    "padding_side": "right",
                    "padding": "max_length",
                    "truncation": True,
                    "tokenizer_name": _TOKENIZER,
                },
            },
            {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": None}},
        ],
    }), encoding="utf-8")
    (cache / "policy_postprocessor.json").write_text(json.dumps({
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": 1e-8,
                    "features": {"action": {"type": "ACTION", "shape": [action_dim]}},
                    "norm_map": norm_map,
                },
                "state_file": post_name,
            },
            {"registry_name": "device_processor", "config": {"device": "cpu", "float_dtype": None}},
        ],
    }), encoding="utf-8")


def _config(
    *,
    chunk_size: int,
    action_dim: int,
    state_dim: int,
    empty_cameras: int,
    num_inference_steps: int,
    n_action_steps: int | None = None,
) -> dict:
    return {
        "type": "pi05",
        "chunk_size": int(chunk_size),
        "n_action_steps": int(n_action_steps) if n_action_steps is not None else int(chunk_size),
        "max_action_dim": 32,
        "max_state_dim": 32,
        "num_inference_steps": int(num_inference_steps),
        "empty_cameras": int(empty_cameras),
        "image_resolution": [224, 224],
        "tokenizer_max_length": 200,
        "tokenizer_name": _TOKENIZER,
        "input_features": {"observation.state": {"type": "STATE", "shape": [int(state_dim)]}},
        "output_features": {"action": {"type": "ACTION", "shape": [int(action_dim)]}},
    }
