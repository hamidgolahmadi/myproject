#!/usr/bin/env python3
"""Run one resumable D047 beta/replication range."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.beta_sweep_production import run_beta_sweep_range
from src.experiments.refined.beta_sweep_protocol import first_beta_sweep_protocol


def _parser() -> argparse.ArgumentParser:
    protocol = first_beta_sweep_protocol()
    parser = argparse.ArgumentParser(
        description="Run one beta slice of the frozen D047 exploratory matched R/SW/SF sweep."
    )
    parser.add_argument("--beta-index", type=int, required=True)
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
        default=REPO_ROOT / "results" / "refined" / "beta_sweep",
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
            "Refusing D047 production execution on an Iridis login node. "
            "Submit the Slurm array wrapper instead."
        )
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")

    protocol = first_beta_sweep_protocol()
    if not 0 <= args.beta_index < protocol.n_beta:
        raise SystemExit(f"--beta-index must lie in [0,{protocol.n_beta - 1}]")
    beta = protocol.beta_grid[args.beta_index]

    def progress(local_index: int, total: int, from_checkpoint: bool) -> None:
        if local_index == 1 or local_index == total or local_index % args.progress_every == 0:
            replication_id = args.start + local_index - 1
            source = "checkpoint" if from_checkpoint else "computed"
            print(
                f"[beta-sweep] {local_index}/{total} beta_index={args.beta_index} "
                f"replication={replication_id} ({source})",
                flush=True,
            )

    print("=== D047 exploratory beta sweep range ===")
    print(f"experiment_seed={protocol.experiment_seed}")
    print(f"alpha_anchor={protocol.alpha_anchor}")
    print(f"beta_index={args.beta_index}")
    print(f"beta={beta}")
    print(f"range=[{args.start}, {args.stop})")
    print(f"outdir={args.outdir}")
    print(f"resume={not args.no_resume}")
    print(f"pid={os.getpid()}")

    records = run_beta_sweep_range(
        beta_index=args.beta_index,
        start_replication=args.start,
        stop_replication=args.stop,
        output_dir=args.outdir,
        resume=not args.no_resume,
        protocol=protocol,
        progress_callback=progress,
    )
    print(
        f"Completed beta slice range with {len(records) // 3} paired replications "
        f"and {len(records)} treatment records."
    )
    print("No beta/topology curve is reported before full D047 finalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
