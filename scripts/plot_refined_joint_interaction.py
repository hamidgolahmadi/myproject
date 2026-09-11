#!/usr/bin/env python3
"""Generate thesis-ready figures from finalized D049 interaction artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.joint_interaction_plotting import (
    generate_joint_interaction_figures,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot finalized D049 joint alpha-beta-gamma_R results without "
            "recomputing simulations or bootstrap statistics."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "joint_interaction",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "joint_interaction" / "figures",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="PNG for inspection and vector PDF for thesis inclusion.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")

    print("=== D049 joint-interaction plotting ===")
    print(f"results_dir={args.results_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"formats={tuple(args.formats)}")
    print(f"dpi={args.dpi}")

    paths = generate_joint_interaction_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        dpi=args.dpi,
    )

    print(f"generated={len(paths)}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
