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
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=8)
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
            str(args.num_samples),
            "--suite",
            args.suite,
            "--seed",
            str(args.seed),
            "--num-steps-wait",
            str(args.num_steps_wait),
            "--state-dim",
            str(args.state_dim),
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


def run_export(args: argparse.Namespace) -> int:
    if args.benchmark != "libero":
        raise ValueError(f"Unsupported benchmark: {args.benchmark!r}")
    cfg = LiberoExportConfig(
        suite=args.suite,
        seed=args.seed,
        env_seed=args.env_seed,
        num_steps_wait=args.num_steps_wait,
        max_tasks=args.max_tasks,
        state_dim=args.state_dim,
    )
    out = export_libero_dataset(
        args.output,
        num_samples=args.num_samples,
        config=cfg,
    )
    print(f"Exported {args.num_samples} samples to {out}")
    return 0


def cmd_export_calibration(args: argparse.Namespace) -> None:
    if not args.no_docker and not _inside_docker():
        _export_via_docker(args)
        return
    sys.exit(run_export(args))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.no_docker and not _inside_docker():
        _export_via_docker(args)
        return 0
    return run_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
