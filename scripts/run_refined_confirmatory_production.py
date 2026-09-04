#!/usr/bin/env python3
"""Run a resumable range of frozen D045 paired confirmatory replications."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.confirmatory_production import run_confirmatory_production_range
from src.experiments.refined.confirmatory_protocol import first_confirmatory_production_protocol


def _parser() -> argparse.ArgumentParser:
    protocol = first_confirmatory_production_protocol()
    parser = argparse.ArgumentParser(
        description="Run a half-open range of the frozen D045 1000-pair confirmatory production sample."
    )
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
        default=REPO_ROOT / "results" / "refined" / "confirmatory_production",
    )
    parser.add_argument("--no-resume", action="store_true", help="Recompute checkpoints in the selected range.")
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
            "Refusing D045 production execution on an Iridis login node. "
            "Submit the Slurm array wrapper instead."
        )
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")

    def progress(local_index: int, total: int, from_checkpoint: bool) -> None:
        if local_index == 1 or local_index == total or local_index % args.progress_every == 0:
            replication_id = args.start + local_index - 1
            source = "checkpoint" if from_checkpoint else "computed"
            print(
                f"[confirmatory] {local_index}/{total} replication={replication_id} ({source})",
                flush=True,
            )

    protocol = first_confirmatory_production_protocol()
    print("=== D045 paired confirmatory production range ===")
    print(f"experiment_seed={protocol.experiment_seed}")
    print(f"range=[{args.start}, {args.stop})")
    print(f"outdir={args.outdir}")
    print(f"resume={not args.no_resume}")
    print(f"pid={os.getpid()}")

    records = run_confirmatory_production_range(
        start_replication=args.start,
        stop_replication=args.stop,
        output_dir=args.outdir,
        resume=not args.no_resume,
        protocol=protocol,
        progress_callback=progress,
    )
    print(
        f"Completed range with {len(records) // 3} paired replications "
        f"and {len(records)} treatment records."
    )
    print("No topology contrasts are reported until the full 1000-pair sample is finalised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
