#!/usr/bin/env python3
"""Finalize the complete frozen D049 joint interaction experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.joint_interaction_production import (
    finalize_joint_interaction_production,
)
from src.experiments.refined.joint_interaction_protocol import (
    first_joint_interaction_protocol,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize the complete frozen D049 joint interaction experiment."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "joint_interaction",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    protocol = first_joint_interaction_protocol()
    print("=== D049 final joint interaction analysis ===")
    print(f"cells={protocol.n_cells}")
    print(f"replications_per_cell={protocol.n_replications}")
    print(f"total_simulations={protocol.n_simulations}")
    print(f"bootstrap_draws={protocol.n_bootstrap}")
    print(f"bootstrap_seed={protocol.bootstrap_seed}")
    print(f"outdir={args.outdir}")
    paths = finalize_joint_interaction_production(output_dir=args.outdir, protocol=protocol)
    print("D049 finalization completed successfully.")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
