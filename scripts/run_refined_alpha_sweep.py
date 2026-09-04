#!/usr/bin/env python3
"""Run one resumable D046 alpha/replication range."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.alpha_sweep_production import run_alpha_sweep_range
from src.experiments.refined.alpha_sweep_protocol import first_alpha_sweep_protocol


def _parser() -> argparse.ArgumentParser:
    protocol = first_alpha_sweep_protocol()
    parser = argparse.ArgumentParser(
        description="Run one alpha slice of the frozen D046 exploratory matched R/SW/SF sweep."
    )
    parser.add_argument("--alpha-index", type=int, required=True)
    parser.add_argument("--start", type=int, default=0, help="First replication id, inclusive.")
    parser.add_argument(
        "--stop",
        type=int,
        default=protocol.n_replications,
        help="Last replication id, exclusive.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "alpha_sweep",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--allow-login-node",
        action="store_true",
        help="Explicitly bypass the Iridis login-node safety guard.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    hostname = socket.gethostname().lower()
    if "login" in hostname and not args.allow_login_node:
        raise SystemExit(
            "Refusing D046 production execution on an Iridis login node. "
            "Submit the Slurm array wrapper instead."
        )
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")

    protocol = first_alpha_sweep_protocol()
    if not 0 <= args.alpha_index < protocol.n_alpha:
        raise SystemExit(
            f"--alpha-index must lie in [0,{protocol.n_alpha - 1}]"
        )
    alpha = protocol.alpha_grid[args.alpha_index]

    def progress(local_index: int, total: int, from_checkpoint: bool) -> None:
        if local_index == 1 or local_index == total or local_index % args.progress_every == 0:
            replication_id = args.start + local_index - 1
            source = "checkpoint" if from_checkpoint else "computed"
            print(
                f"[alpha-sweep] {local_index}/{total} alpha_index={args.alpha_index} "
                f"replication={replication_id} ({source})",
                flush=True,
            )

    print("=== D046 exploratory alpha sweep range ===")
    print(f"experiment_seed={protocol.experiment_seed}")
    print(f"alpha_index={args.alpha_index}")
    print(f"alpha={alpha}")
    print(f"range=[{args.start}, {args.stop})")
    print(f"outdir={args.outdir}")
    print(f"resume={not args.no_resume}")
    print(f"pid={os.getpid()}")

    records = run_alpha_sweep_range(
        alpha_index=args.alpha_index,
        start_replication=args.start,
        stop_replication=args.stop,
        output_dir=args.outdir,
        resume=not args.no_resume,
        protocol=protocol,
        progress_callback=progress,
    )
    print(
        f"Completed alpha slice range with {len(records) // 3} paired replications "
        f"and {len(records)} treatment records."
    )
    print("No alpha/topology curve is reported before full D046 finalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
