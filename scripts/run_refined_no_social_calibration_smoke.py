#!/usr/bin/env python3
"""Run the small seed-disjoint end-to-end D042 calibration smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.refined.calibration_smoke import (
    NoSocialCalibrationSmokeProtocol,
    run_no_social_calibration_smoke,
    write_no_social_calibration_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/refined/no_social_calibration_smoke"),
    )
    parser.add_argument(
        "--n-scale-replications",
        type=int,
        default=3,
        help="Smoke-only count; does not alter final D042 count of 500.",
    )
    parser.add_argument(
        "--n-threshold-replications",
        type=int,
        default=3,
        help="Smoke-only count; does not alter final D042 count of 500.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    protocol = NoSocialCalibrationSmokeProtocol(
        n_scale_replications=args.n_scale_replications,
        n_threshold_replications=args.n_threshold_replications,
    )
    result = run_no_social_calibration_smoke(smoke_protocol=protocol)
    artifact = write_no_social_calibration_smoke(result, outdir=args.outdir)

    scales = result.calibration.reference_scales
    print("Refined no-social calibration smoke completed")
    print(
        f"alpha={result.market_protocol.calibration_alpha:g}, "
        f"scale runs={protocol.n_scale_replications}, "
        f"threshold runs={protocol.n_threshold_replications}, "
        f"rolling points/run={result.market_protocol.expected_rolling_points_per_run}"
    )
    print("Purpose: pipeline validation only; these are NOT final D042 calibration values.")
    print()
    print(f"c_ret = {scales.return_scale:.10g}")
    print(f"c_bel = {scales.belief_scale:.10g}")
    print(f"c_F   = {scales.order_flow_scale:.10g}")
    print(f"c_CID = {result.calibration.cid_threshold:.10g}")
    print("threshold peak CIDs:", ", ".join(f"{value:.10g}" for value in result.threshold_peak_cids))
    print()
    print(f"artifact: {artifact.resolve()}")


if __name__ == "__main__":
    main()
