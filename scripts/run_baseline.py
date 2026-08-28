"""
Command-line runner for fixed-network baseline experiments.

This script is the external entry point for running repeated baseline
simulations on pre-generated network topologies.

Responsibilities
----------------
1. Parse command-line options.
2. Load pre-generated topology matrices.
3. Run the baseline experiment across simulation seeds.
4. Aggregate run-level results by topology.
5. Save raw, run-level, aggregate, and configuration outputs.

The economic model itself lives in ``src/model``.
Experimental logic lives in ``src/experiments``.
Aggregation logic lives in ``src/analysis``.
Network loading utilities live in ``src/topologies``.

This separation keeps the command-line script intentionally thin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# Project Path Setup
# =============================================================================

# This script lives in:
#
#     project_root/scripts/run_baseline.py
#
# Add the project root explicitly so direct execution such as
#
#     python3 scripts/run_baseline.py
#
# can reliably import modules from ``src``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# Project Imports
# =============================================================================

from src.analysis.aggregate_results import (
    aggregate_baseline_by_topology,
)

from src.experiments.baseline import (
    DEFAULT_STABILITY_THRESHOLDS,
    run_baseline_batch,
)

from src.topologies.topology_utils import (
    load_topology,
)


# =============================================================================
# Topology Naming
# =============================================================================

def topology_names_for_mode(
    mode: str,
) -> list[str]:
    """
    Return the topology classes associated with a generation mode.

    Parameters
    ----------
    mode : str
        Either ``standard`` or ``extreme``.

    Returns
    -------
    list[str]
        Expected topology directory names.
    """

    if mode == "standard":
        return [
            "random_fixed",
            "small_world",
            "scale_free",
        ]

    if mode == "extreme":
        return [
            "random_fixed",
            "small_world_clustered",
            "scale_free_extreme",
        ]

    raise ValueError(
        f"Unknown topology mode: {mode}"
    )


# =============================================================================
# Loading One Topology Realisation per Class
# =============================================================================

def load_topologies(
    topology_root: Path,
    mode: str,
    topology_seed: int,
) -> dict:
    """
    Load one pre-generated topology realisation for each topology class.

    Parameters
    ----------
    topology_root : pathlib.Path
        Root directory containing the topology bank.

    mode : str
        ``standard`` or ``extreme``.

    topology_seed : int
        Seed identifying the network realisation to load.

    Returns
    -------
    dict
        Mapping from topology name to influence matrix.
    """

    topology_names = topology_names_for_mode(
        mode
    )

    topologies = {}

    for topology_name in topology_names:

        topology_path = (
            topology_root
            / mode
            / topology_name
            / f"seed_{topology_seed:06d}.npz"
        )

        if not topology_path.exists():
            raise FileNotFoundError(
                f"Missing topology file: {topology_path}"
            )

        topologies[topology_name] = load_topology(
            topology_path
        )

    return topologies


# =============================================================================
# Command-Line Arguments
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options for the baseline experiment.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run repeated fixed-network baseline simulations "
            "on pre-generated topology matrices."
        )
    )

    parser.add_argument(
        "--topology-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "topology_cache"
        ),
    )

    parser.add_argument(
        "--topology-mode",
        choices=[
            "standard",
            "extreme",
        ],
        default="extreme",
    )

    parser.add_argument(
        "--topology-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--riskS-mean-max",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLDS[
            "riskS_mean_max"
        ],
    )

    parser.add_argument(
        "--ret-vol-max",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLDS[
            "ret_vol_max"
        ],
    )

    parser.add_argument(
        "--belief-var-max",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLDS[
            "belief_var_max"
        ],
    )

    parser.add_argument(
        "--flow-std-max",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLDS[
            "flow_std_max"
        ],
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_STABILITY_THRESHOLDS[
            "window"
        ],
    )

    parser.add_argument(
        "--required-consecutive-windows",
        type=int,
        default=DEFAULT_STABILITY_THRESHOLDS[
            "required_consecutive_windows"
        ],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "baseline"
        ),
    )

    return parser.parse_args()


# =============================================================================
# Argument Validation
# =============================================================================

def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """
    Validate basic experiment controls before computation begins.
    """

    if args.horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    if args.n_seeds <= 0:
        raise ValueError(
            "n-seeds must be positive."
        )

    if args.topology_seed < 0:
        raise ValueError(
            "topology-seed cannot be negative."
        )

    if args.window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if args.required_consecutive_windows <= 0:
        raise ValueError(
            "required-consecutive-windows must be positive."
        )


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    """
    Run the complete baseline experiment from the command line.
    """

    args = parse_arguments()

    validate_arguments(
        args
    )

    # -------------------------------------------------------------------------
    # Stability configuration
    # -------------------------------------------------------------------------

    threshold_cfg = {
        "riskS_mean_max": args.riskS_mean_max,
        "ret_vol_max": args.ret_vol_max,
        "belief_var_max": args.belief_var_max,
        "flow_std_max": args.flow_std_max,
        "window": args.window,
        "required_consecutive_windows": (
            args.required_consecutive_windows
        ),
    }

    # -------------------------------------------------------------------------
    # Load topology matrices
    # -------------------------------------------------------------------------

    print("Loading topology realisations...")

    topologies = load_topologies(
        topology_root=args.topology_root,
        mode=args.topology_mode,
        topology_seed=args.topology_seed,
    )

    print(
        "Loaded:",
        ", ".join(topologies.keys()),
    )

    # -------------------------------------------------------------------------
    # Run simulations
    # -------------------------------------------------------------------------

    print()
    print(
        f"Running {args.n_seeds} simulation seeds "
        f"per topology with horizon={args.horizon}..."
    )

    raw_df, summary_df = run_baseline_batch(
        topologies=topologies,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        horizon=args.horizon,
        threshold_cfg=threshold_cfg,
    )

    # -------------------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------------------

    aggregate_df = aggregate_baseline_by_topology(
        summary_df
    )

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "baseline_raw.csv"
    )

    summary_path = (
        output_dir
        / "baseline_summary_by_seed.csv"
    )

    aggregate_path = (
        output_dir
        / "baseline_summary.csv"
    )

    threshold_path = (
        output_dir
        / "stability_thresholds.csv"
    )

    run_config_path = (
        output_dir
        / "run_configuration.csv"
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    pd.DataFrame(
        [threshold_cfg]
    ).to_csv(
        threshold_path,
        index=False,
    )

    run_configuration = {
        "topology_mode": args.topology_mode,
        "topology_seed": args.topology_seed,
        "horizon": args.horizon,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
    }

    pd.DataFrame(
        [run_configuration]
    ).to_csv(
        run_config_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print()
    print(
        "Baseline experiment completed successfully."
    )

    print()
    print("Saved:")
    print(raw_path)
    print(summary_path)
    print(aggregate_path)
    print(threshold_path)
    print(run_config_path)

    print()
    print(
        aggregate_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
