#!/usr/bin/env python3
"""Run the small paired R/SW/SF confirmatory pipeline smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.refined.confirmatory_runner import (  # noqa: E402
    run_paired_confirmatory_smoke,
    write_paired_confirmatory_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replications",
        type=int,
        default=2,
        help="Number of paired smoke replications (default: 2).",
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=2026090401,
        help="Smoke-only seed namespace; not for final confirmatory Monte Carlo.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "results" / "refined" / "confirmatory_smoke",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_paired_confirmatory_smoke(
        experiment_seed=args.experiment_seed,
        n_replications=args.replications,
    )
    records_path, metadata_path = write_paired_confirmatory_smoke(
        result,
        outdir=args.outdir,
    )

    print("Paired confirmatory smoke completed")
    print(
        f"paired replications={result.n_replications}, records={len(result.records)}, "
        "topologies=R/SW/SF, regimes=baseline+alpha0"
    )
    print("Purpose: pipeline validation only; do not rank topologies from this smoke.")

    for replication_id in range(result.n_replications):
        alpha0 = [
            record
            for record in result.records
            if record.replication_id == replication_id and record.regime == "alpha0_control"
        ]
        fingerprints = {record.economic_path_fingerprint for record in alpha0}
        print(
            f"alpha0 replication {replication_id}: "
            f"economic-path fingerprints across R/SW/SF = {len(fingerprints)} unique"
        )

    print(f"records: {records_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
