#!/usr/bin/env python3
"""Run/resume the final frozen-method D042 no-social calibration."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.refined.market_calibration_run import (  # noqa: E402
    load_reference_scales,
    run_final_market_calibration,
    run_scale_calibration_stage,
    run_threshold_calibration_stage,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("scale", "threshold", "all"),
        default="all",
        help="Run only the scale stage, only threshold stage, or both in order.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "results" / "refined" / "market_calibration",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute stage checkpoints instead of reusing valid existing checkpoints.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N replications (default: 10).",
    )
    parser.add_argument(
        "--allow-login-node",
        action="store_true",
        help="Override the Iridis login-node safety check. Not recommended for the 500+500 run.",
    )
    return parser


def _on_login_node() -> bool:
    host = socket.gethostname().lower()
    return "login" in host


def main() -> None:
    args = _parser().parse_args()
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be a positive integer")
    if _on_login_node() and not args.allow_login_node:
        raise SystemExit(
            "Refusing to run the 500+500 production calibration on an Iridis login node. "
            "Submit this script inside a compute allocation/batch job, or use "
            "--allow-login-node only if you deliberately accept that risk."
        )

    def progress(stage: str, completed: int, total: int, reused: bool) -> None:
        if completed == 1 or completed == total or completed % args.progress_every == 0:
            status = "checkpoint" if reused else "computed"
            print(f"[{stage}] {completed}/{total} ({status})", flush=True)

    resume = not args.no_resume
    if args.stage == "scale":
        scales = run_scale_calibration_stage(
            output_dir=args.outdir,
            resume=resume,
            progress=progress,
        )
        print("D042 scale stage complete")
        print(f"c_ret = {scales.return_scale:.12g}")
        print(f"c_bel = {scales.belief_scale:.12g}")
        print(f"c_F   = {scales.order_flow_scale:.12g}")
        print(f"artifact: {(args.outdir / 'reference_scales.json').resolve()}")
        return

    if args.stage == "threshold":
        scales = load_reference_scales(args.outdir)
        result = run_threshold_calibration_stage(
            output_dir=args.outdir,
            scales=scales,
            resume=resume,
            progress=progress,
        )
    else:
        result = run_final_market_calibration(
            output_dir=args.outdir,
            resume=resume,
            progress=progress,
        )

    scales = result.calibration.reference_scales
    print("Final D042 no-social calibration completed")
    print(f"c_ret = {scales.return_scale:.12g}")
    print(f"c_bel = {scales.belief_scale:.12g}")
    print(f"c_F   = {scales.order_flow_scale:.12g}")
    print(f"c_CID = {result.calibration.cid_threshold:.12g}")
    print(f"artifact: {(args.outdir / 'market_evaluation_calibration.json').resolve()}")


if __name__ == "__main__":
    main()
