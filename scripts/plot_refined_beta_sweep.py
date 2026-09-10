#!/usr/bin/env python3
"""Generate thesis-ready figures from the finalized D047 beta-sweep artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.refined.beta_sweep_plotting import generate_beta_sweep_figures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot finalized D047 beta-sweep results without recomputing any simulation "
            "or bootstrap statistic."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "beta_sweep",
        help="Directory containing finalized D047 CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "refined" / "beta_sweep" / "figures",
        help="Figure destination. The default stays inside ignored results/.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("grid", "log", "symlog"),
        default=("symlog",),
        help=(
            "symlog = preferred thesis view retaining beta=0 while resolving the full range; "
            "grid = equally spaced predeclared beta values; "
            "log = positive beta values only on a logarithmic axis."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Output formats. PDF is vector and suitable for the thesis.",
    )
    parser.add_argument(
        "--include-detail",
        action="store_true",
        help="Also create CI plots for the core market and mechanism relative gaps.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution; ignored for vector PDF output.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")

    print("=== D047 beta-sweep plotting ===")
    print(f"results_dir={args.results_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"modes={tuple(args.modes)}")
    print(f"formats={tuple(args.formats)}")
    print(f"include_detail={args.include_detail}")

    paths = generate_beta_sweep_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        modes=tuple(args.modes),
        formats=tuple(args.formats),
        include_detail=args.include_detail,
        dpi=args.dpi,
    )

    print(f"generated={len(paths)}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
