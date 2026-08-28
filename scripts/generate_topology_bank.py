"""
Generate a reproducible bank of network topologies.

This script repeatedly calls the topology generators defined in
``src/topologies`` and stores the resulting influence matrices on disk.

For each replication (seed), the script can generate:

    - Random fixed-degree networks
    - Small-world networks
    - Scale-free networks

Two topology modes are supported:

    standard
        Uses the regular benchmark topology generators.

    extreme
        Uses deliberately more distinct small-world and scale-free structures.
        This mode is useful for experiments designed to maximise structural
        separation between topology classes.

The script also computes structural diagnostics for every generated network
and writes them into a single CSV summary file.

Example
-------
Generate 1,000 standard networks of each topology type:

    python3 scripts/generate_topology_bank.py \
        --mode standard \
        --n 100 \
        --k 6 \
        --n-seeds 1000

Generate 1,000 extreme networks:

    python3 scripts/generate_topology_bank.py \
        --mode extreme \
        --n 100 \
        --k 6 \
        --n-seeds 1000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# =============================================================================
# Project Path Setup
# =============================================================================

# ``generate_topology_bank.py`` lives inside the ``scripts`` directory.
# When a Python file is executed directly, Python normally searches for modules
# relative to that directory. We therefore add the project root explicitly so
# that imports such as ``src.topologies`` work reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Project Imports
# =============================================================================

from src.metrics.network_metrics import topology_metrics

from src.topologies.random_network import build_P_random_fixed
from src.topologies.scale_free import build_P_scale_free
from src.topologies.small_world import build_P_small_world

from src.topologies.extreme_topologies import (
    build_scale_free_extreme,
    build_small_world_clustered,
)

from src.topologies.topology_utils import (
    normalize_rows,
    save_topology,
)


# =============================================================================
# Input Validation
# =============================================================================

def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments before topology generation begins.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Raises
    ------
    ValueError
        If an invalid network size, degree, seed count, or rewiring
        probability is supplied.
    """

    if args.n <= 1:
        raise ValueError(
            "Network size 'n' must be greater than 1."
        )

    if args.k <= 0:
        raise ValueError(
            "Number of links 'k' must be positive."
        )

    # Because self-links are excluded, each node can connect to at most n - 1
    # other nodes.
    if args.k >= args.n:
        raise ValueError(
            "'k' must be smaller than 'n' because self-links are excluded."
        )

    if args.n_seeds <= 0:
        raise ValueError(
            "'n-seeds' must be positive."
        )

    if not 0.0 <= args.beta <= 1.0:
        raise ValueError(
            "'beta' must lie between 0 and 1."
        )


# =============================================================================
# Standard Topology Generation
# =============================================================================

def generate_standard_topologies(
    n: int,
    k: int,
    beta: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Generate the three standard benchmark topology classes.

    A deterministic transformation of the replication seed is used for each
    topology class. This ensures that:

        1. Results are fully reproducible.
        2. Different topology classes do not consume identical random streams.

    Parameters
    ----------
    n : int
        Number of agents/nodes.

    k : int
        Number of neighbours assigned to each node.

    beta : float
        Rewiring probability for the small-world topology.

    seed : int
        Replication-level random seed.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from topology name to weighted influence matrix.
    """

    # Use separate deterministic seeds for the three topology classes.
    random_seed = seed * 10 + 1
    small_world_seed = seed * 10 + 2
    scale_free_seed = seed * 10 + 3

    P_random = build_P_random_fixed(
        N=n,
        K=k,
        seed=random_seed,
    )

    P_small_world = build_P_small_world(
        N=n,
        K=k,
        beta=beta,
        seed=small_world_seed,
    )

    P_scale_free = build_P_scale_free(
        N=n,
        K=k,
        seed=scale_free_seed,
    )

    return {
        "random_fixed": P_random,
        "small_world": P_small_world,
        "scale_free": P_scale_free,
    }


# =============================================================================
# Extreme Topology Generation
# =============================================================================

def generate_extreme_topologies(
    n: int,
    k: int,
    beta: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Generate deliberately separated topology classes.

    The random network remains the benchmark random-fixed topology, while the
    small-world and scale-free generators use the stronger structural designs
    defined in ``extreme_topologies.py``.

    Parameters
    ----------
    n : int
        Number of agents/nodes.

    k : int
        Number of neighbours assigned to each node.

    beta : float
        Rewiring probability for the clustered small-world topology.

    seed : int
        Replication-level random seed.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from topology name to weighted influence matrix.
    """

    # Give each topology an independent but reproducible random stream.
    random_seed = seed * 10 + 1

    rng_small_world = np.random.default_rng(
        seed * 10 + 2
    )

    rng_scale_free = np.random.default_rng(
        seed * 10 + 3
    )

    # The random topology provides a neutral benchmark.
    P_random = build_P_random_fixed(
        N=n,
        K=k,
        seed=random_seed,
    )

    # Strongly clustered small-world topology.
    P_small_world = build_small_world_clustered(
        n=n,
        k=k,
        rng=rng_small_world,
        beta=beta,
    )

    # Strongly hub-dominated scale-free-style topology.
    P_scale_free = build_scale_free_extreme(
        n=n,
        k=k,
        rng=rng_scale_free,
    )

    return {
        "random_fixed": P_random,
        "small_world_clustered": P_small_world,
        "scale_free_extreme": P_scale_free,
    }


# =============================================================================
# Saving One Replication
# =============================================================================

def save_replication(
    topologies: dict[str, np.ndarray],
    seed: int,
    output_directory: Path,
    mode: str,
) -> list[dict]:
    """
    Save all topology matrices produced for one replication.

    Structural network diagnostics are also calculated and returned so that
    they can later be combined into a single summary table.

    Parameters
    ----------
    topologies : dict[str, np.ndarray]
        Mapping from topology names to influence matrices.

    seed : int
        Replication seed.

    output_directory : pathlib.Path
        Root directory for the generated topology bank.

    mode : str
        Topology generation mode: ``standard`` or ``extreme``.

    Returns
    -------
    list[dict]
        Structural summary rows for the generated networks.
    """

    summary_rows = []

    for topology_name, P in topologies.items():

        # Enforce row normalisation before the topology is stored.
        # The current generators already produce row-stochastic matrices, but
        # this provides an additional safeguard against numerical deviations.
        P = normalize_rows(P)

        # Keep different topology classes in separate directories.
        topology_directory = (
            output_directory
            / mode
            / topology_name
        )

        # Each generated network has a unique filename determined by its seed.
        topology_path = (
            topology_directory
            / f"seed_{seed:06d}.npz"
        )

        # Save the weighted influence matrix.
        save_topology(
            topology_path,
            P,
        )

        # Compute structural diagnostics for later inspection.
        metrics = topology_metrics(
            topology_name,
            P,
        )

        # Add experiment-identification information to the metric record.
        metrics["seed"] = seed
        metrics["mode"] = mode
        metrics["file"] = str(
            topology_path.relative_to(PROJECT_ROOT)
        )

        summary_rows.append(metrics)

    return summary_rows


# =============================================================================
# Summary Output
# =============================================================================

def save_summary_csv(
    rows: list[dict],
    path: Path,
) -> None:
    """
    Save structural diagnostics for the complete topology bank.

    Parameters
    ----------
    rows : list[dict]
        One dictionary per generated topology.

    path : pathlib.Path
        Destination CSV file.
    """

    if not rows:
        raise ValueError(
            "No topology summary rows were generated."
        )

    # Ensure that the destination directory exists.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use the keys of the first record as CSV column names.
    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Command-Line Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed user options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate repeated random network topologies "
            "for simulation experiments."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["standard", "extreme"],
        default="standard",
        help=(
            "Topology design to generate. "
            "Default: standard."
        ),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help=(
            "Number of agents/nodes. "
            "Default: 100."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help=(
            "Number of neighbours per node. "
            "Default: 6."
        ),
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help=(
            "Small-world rewiring probability. "
            "Default: 0.02."
        ),
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help=(
            "First replication seed. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=100,
        help=(
            "Number of independent topology replications. "
            "Default: 100."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "topology_cache",
        help=(
            "Root directory in which generated topology matrices "
            "will be stored."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    """
    Generate the requested topology bank and structural summary.
    """

    args = parse_arguments()

    # Fail early when command-line parameters are invalid.
    validate_arguments(args)

    output_directory = args.output_dir.resolve()

    # Store all structural diagnostics here before writing the final CSV file.
    all_summary_rows = []

    seed_stop = (
        args.seed_start
        + args.n_seeds
    )

    print(
        f"Generating {args.n_seeds} replications "
        f"in '{args.mode}' mode..."
    )

    # -------------------------------------------------------------------------
    # Main replication loop
    # -------------------------------------------------------------------------

    for seed in range(
        args.seed_start,
        seed_stop,
    ):

        # Select the appropriate topology construction regime.
        if args.mode == "standard":

            topologies = generate_standard_topologies(
                n=args.n,
                k=args.k,
                beta=args.beta,
                seed=seed,
            )

        else:

            topologies = generate_extreme_topologies(
                n=args.n,
                k=args.k,
                beta=args.beta,
                seed=seed,
            )

        # Save matrices and collect their structural diagnostics.
        rows = save_replication(
            topologies=topologies,
            seed=seed,
            output_directory=output_directory,
            mode=args.mode,
        )

        all_summary_rows.extend(rows)

        # Print occasional progress information for long Iridis jobs.
        replication_number = (
            seed
            - args.seed_start
            + 1
        )

        if (
            replication_number == 1
            or replication_number % 100 == 0
            or replication_number == args.n_seeds
        ):
            print(
                f"Completed "
                f"{replication_number}/{args.n_seeds} "
                f"replications."
            )

    # -------------------------------------------------------------------------
    # Save structural summary
    # -------------------------------------------------------------------------

    summary_path = (
        output_directory
        / args.mode
        / "topology_structure_summary.csv"
    )

    save_summary_csv(
        rows=all_summary_rows,
        path=summary_path,
    )

    print()
    print("Topology generation completed successfully.")
    print(f"Topology bank: {output_directory / args.mode}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
