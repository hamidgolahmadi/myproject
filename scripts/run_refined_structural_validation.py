#!/usr/bin/env python3
"""Run the first refined structural-only benchmark validation.

With no scientific overrides this script executes Decision D041.  It does not
run the market model and does not generate shocks or attention paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.refined import (  # noqa: E402
    first_structural_validation_calibration,
    run_structural_ensemble,
    write_structural_result,
)


def build_parser() -> argparse.ArgumentParser:
    calibration = first_structural_validation_calibration()
    parser = argparse.ArgumentParser(
        description="Run the refined structural-only R/SW/SF ensemble validation."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/refined/structural_validation"),
        help="Directory for raw CSV, summary CSV, and metadata JSON.",
    )
    parser.add_argument(
        "--n-replications",
        type=int,
        default=calibration.n_replications,
        help=(
            "Number of graph replications per topology. Default is the frozen "
            "D041 value; lower values are intended only for smoke runs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibration = first_structural_validation_calibration()

    result = run_structural_ensemble(
        experiment_seed=calibration.experiment_seed,
        n_replications=args.n_replications,
        n_agents=calibration.n_agents,
        q=calibration.hub_q,
        specifications=calibration.topology_specifications(),
    )
    raw_path, summary_path, metadata_path = write_structural_result(
        result,
        output_directory=args.outdir,
    )

    print("Refined structural validation completed")
    print(
        f"N={calibration.n_agents}, K={calibration.k}, "
        f"replications/topology={args.n_replications}, q={calibration.hub_q}, "
        f"p_sw={calibration.p_sw}, a0={calibration.a0}, "
        f"experiment_seed={calibration.experiment_seed}"
    )
    print()
    print(
        "topology  gini_mean  hub_share_mean  clustering_mean  "
        "apl_lcc_mean  lcc_share_mean"
    )
    for label in result.topology_labels:
        summary = result.summary_for(label)
        print(
            f"{label:<8}  "
            f"{summary.in_degree_gini.mean:9.5f}  "
            f"{summary.hub_link_share.mean:14.5f}  "
            f"{summary.global_clustering.mean:15.5f}  "
            f"{summary.average_path_length_lcc.mean:12.5f}  "
            f"{summary.largest_component_share.mean:14.5f}"
        )

    print()
    print(f"raw records: {raw_path}")
    print(f"summary:     {summary_path}")
    print(f"metadata:    {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
