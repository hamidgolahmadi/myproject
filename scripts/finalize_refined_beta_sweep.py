#!/usr/bin/env python3
"""Finalize D047 only after every beta/replication checkpoint exists."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.beta_sweep_production import finalize_beta_sweep_production
from src.experiments.refined.beta_sweep_protocol import first_beta_sweep_protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create final D047 exploratory beta-sweep curve artifacts."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "beta_sweep",
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
            "Refusing D047 finalization on an Iridis login node. "
            "Submit the finalization Slurm wrapper instead."
        )

    protocol = first_beta_sweep_protocol()
    print("=== D047 final exploratory beta-sweep analysis ===")
    print(f"alpha_anchor={protocol.alpha_anchor}")
    print(f"beta_grid={protocol.beta_grid}")
    print(f"replications_per_beta={protocol.n_replications}")
    print(f"bootstrap_draws={protocol.n_bootstrap}")
    print(f"bootstrap_seed={protocol.bootstrap_seed}")
    print(f"outdir={args.outdir}")

    paths = finalize_beta_sweep_production(
        output_dir=args.outdir,
        protocol=protocol,
    )
    print("D047 finalization completed successfully.")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
