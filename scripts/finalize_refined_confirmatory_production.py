#!/usr/bin/env python3
"""Finalize D045 only after every predeclared paired replication exists."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.confirmatory_production import finalize_confirmatory_production
from src.experiments.refined.confirmatory_protocol import first_confirmatory_production_protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create final D045 bootstrap inference artifacts from the complete 1000-pair sample."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "confirmatory_production",
    )
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
            "Refusing D045 finalization on an Iridis login node. "
            "Submit the finalization Slurm wrapper instead."
        )

    protocol = first_confirmatory_production_protocol()
    print("=== D045 final confirmatory inference ===")
    print(f"required_paired_replications={protocol.n_replications}")
    print(f"bootstrap_draws={protocol.n_bootstrap}")
    print(f"bootstrap_seed={protocol.bootstrap_seed}")
    print(f"outdir={args.outdir}")

    paths = finalize_confirmatory_production(
        output_dir=args.outdir,
        protocol=protocol,
    )
    print("D045 finalization completed successfully.")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
