"""``vla-eval export-calibration`` — run locally or via LIBERO Docker."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from vla_eval.calibration.libero import LiberoExportConfig, export_libero_dataset

logger = logging.getLogger(__name__)

LIBERO_DOCKER_IMAGE = "ghcr.io/allenai/vla-evaluation-harness/libero:latest"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export benchmark observations for offline model calibration.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("libero",),
        default="libero",
        help="Benchmark to sample (default: libero).",
    )
    parser.add_argument("--output", "-o", required=True, help="Output .npz path.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="LIBERO datasets root directory (contains <suite>/<task>_demo.hdf5).",
    )
    parser.add_argument(
        "--rollout-dir",
        default=None,
        help="Sample the calibration npz from FP rollouts saved by eval config rollout_dir.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="With --rollout-dir, keep only episodes whose eval result was success.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Output sample count. Rollout export defaults to 120 and splits "
            "task→prompt→phase, or prompt→phase when prompts cross task names. "
            "LIBERO demo export defaults to 30."
        ),
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=50)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Demo frame grid stride (default 50, matches common replan chunk).",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=2,
        help="Per-anchor uniform jitter in ±jitter action steps (default 2).",
    )
    parser.add_argument(
        "--end-margin",
        type=int,
        default=10,
        help="Drop frames with index >= traj_len - end_margin (default 10).",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run in the current environment (requires LIBERO installed).",
    )
    parser.add_argument(
        "--docker-image",
        default=LIBERO_DOCKER_IMAGE,
        help=f"Docker image when running via container (default: {LIBERO_DOCKER_IMAGE}).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Mount local vla-evaluation-harness src/ into the export container.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip docker pull prompt.")
    return parser


def _inside_docker() -> bool:
    return Path("/.dockerenv").exists()


def _resolve_harness_src() -> Path:
    cwd_src = Path.cwd() / "src"
    if (cwd_src / "vla_eval").is_dir():
        return cwd_src.resolve()
    import vla_eval

    pkg_parent = Path(vla_eval.__file__).resolve().parent.parent
    if pkg_parent.name == "src" and (pkg_parent / "vla_eval").is_dir():
        return pkg_parent
    raise RuntimeError(
        "Cannot locate vla_eval sources for --dev. Run from the "
        "vla-evaluation-harness checkout or use an editable install."
    )


def _export_via_docker(args: argparse.Namespace) -> None:
    from vla_eval.cli._docker import check_docker_daemon, ensure_image_local

    docker = shutil.which("docker")
    if docker is None:
        print("ERROR: docker not found.", file=sys.stderr)
        sys.exit(1)
    check_docker_daemon(docker)
    ensure_image_local(docker, args.docker_image, args.yes)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"LIBERO dataset directory does not exist: {dataset_dir}")

    container_name = f"vla-eval-export-{os.getpid()}"
    cmd: list[str] = [
        docker,
        "run",
        "--rm",
        "--name",
        container_name,
        "--entrypoint",
        "conda",
        "-v",
        f"{output.parent}:{output.parent}",
        "-v",
        f"{dataset_dir}:{dataset_dir}:ro",
    ]
    if args.dev:
        src = _resolve_harness_src()
        cmd.extend(["-v", f"{src}:/workspace/src:ro", "-e", "PYTHONPATH=/workspace/src"])
    cmd.extend(
        [
            args.docker_image,
            "run",
            "--no-capture-output",
            "-n",
            "libero",
            "python",
            "-m",
            "vla_eval.cli.export_calibration",
            "--benchmark",
            args.benchmark,
            "--no-docker",
            "-o",
            str(output),
            "--num-samples",
            str(30 if args.num_samples is None else args.num_samples),
            "--dataset-dir",
            str(dataset_dir),
            "--suite",
            args.suite,
            "--seed",
            str(args.seed),
            "--max-episodes-per-task",
            str(args.max_episodes_per_task),
            "--state-dim",
            str(args.state_dim),
            "--chunk-size",
            str(args.chunk_size),
            "--jitter",
            str(args.jitter),
            "--end-margin",
            str(args.end_margin),
        ]
    )
    if args.env_seed is not None:
        cmd.extend(["--env-seed", str(args.env_seed)])
    if args.max_tasks is not None:
        cmd.extend(["--max-tasks", str(args.max_tasks)])

    logger.info("Running export via Docker: %s", " ".join(cmd))
    import subprocess

    rc = subprocess.call(cmd)
    sys.exit(rc)


def _export_from_rollouts(args: argparse.Namespace) -> int:
    from vla_eval.calibration.rollout import export_rollout_calibration

    num_samples = 120 if args.num_samples is None else args.num_samples
    out = export_rollout_calibration(
        args.rollout_dir,
        args.output,
        num_samples=num_samples,
        seed=args.seed,
        chunk_size=args.chunk_size,
        jitter=args.jitter,
        end_margin=args.end_margin,
        success_only=bool(getattr(args, "success_only", False)),
        state_dim=args.state_dim,
    )
    print(f"Exported {num_samples} samples to {out}")
    return 0


def run_export(args: argparse.Namespace) -> int:
    if not args.dataset_dir:
        raise ValueError("LIBERO export requires --dataset-dir, or pass --rollout-dir to sample saved rollouts.")
    if args.benchmark != "libero":
        raise ValueError(f"Unsupported benchmark: {args.benchmark!r}")
    cfg = LiberoExportConfig(
        dataset_dir=args.dataset_dir,
        suite=args.suite,
        seed=args.seed,
        env_seed=args.env_seed,
        max_episodes_per_task=args.max_episodes_per_task,
        max_tasks=args.max_tasks,
        state_dim=args.state_dim,
        chunk_size=args.chunk_size,
        jitter=args.jitter,
        end_margin=args.end_margin,
    )
    num_samples = 30 if args.num_samples is None else args.num_samples
    out = export_libero_dataset(
        args.output,
        num_samples=num_samples,
        config=cfg,
    )
    print(f"Exported {num_samples} samples to {out}")
    return 0


def cmd_export_calibration(args: argparse.Namespace) -> None:
    if getattr(args, "rollout_dir", None):
        sys.exit(_export_from_rollouts(args))
    if not args.no_docker and not _inside_docker():
        _export_via_docker(args)
        return
    sys.exit(run_export(args))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rollout_dir:
        return _export_from_rollouts(args)
    if not args.no_docker and not _inside_docker():
        _export_via_docker(args)
        return 0
    return run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
