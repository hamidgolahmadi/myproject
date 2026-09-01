"""
Fixed-network baseline experiment runner.

This module defines the experimental logic for running the market model on
one or more fixed network topologies.

Responsibilities of this module
-------------------------------
- Run one simulation on one topology.
- Compute period-level stability classifications.
- Compute run-level summary statistics.
- Repeat the experiment across topology classes and random seeds.

This module deliberately does NOT:
- parse command-line arguments,
- load or save files,
- aggregate results across replications for publication tables,
- create figures.

Those responsibilities belong to the scripts, analysis, and plotting layers.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.metrics.market_metrics import summarize_run
from src.metrics.stability_metrics import (
    rolling_stability_flags,
    time_to_stability,
)
from src.model.baseline_env import FixedNetworkMarketEnv


# =============================================================================
# Default Stability Configuration
# =============================================================================

DEFAULT_STABILITY_THRESHOLDS = {
    "riskS_mean_max": 2.0,
    "ret_vol_max": 0.0020,
    "belief_var_max": 0.015,
    "flow_std_max": 0.12,
    "window": 50,
    "required_consecutive_windows": 3,
}


# =============================================================================
# Stability-Configuration Validation
# =============================================================================

def validate_threshold_config(
    threshold_cfg: Mapping,
) -> None:
    """
    Check that all stability parameters required by the experiment are present.

    Parameters
    ----------
    threshold_cfg : Mapping
        Stability-threshold configuration.

    Raises
    ------
    KeyError
        If a required parameter is missing.
    """

    required_keys = {
        "riskS_mean_max",
        "ret_vol_max",
        "belief_var_max",
        "flow_std_max",
        "window",
        "required_consecutive_windows",
    }

    missing = (
        required_keys
        - set(threshold_cfg)
    )

    if missing:
        raise KeyError(
            "Missing stability parameters: "
            + ", ".join(sorted(missing))
        )


# =============================================================================
# One Topology / One Seed
# =============================================================================

def run_one_topology(
    topology_name: str,
    P: np.ndarray,
    seed: int,
    horizon: int,
    threshold_cfg: Mapping | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run one complete baseline simulation.

    Parameters
    ----------
    topology_name : str
        Name identifying the network topology.

    P : np.ndarray
        Fixed row-stochastic influence matrix.

    seed : int
        Random seed controlling the stochastic simulation path.

    horizon : int
        Number of simulation periods.

    threshold_cfg : Mapping or None
        Stability-threshold configuration. When omitted, the project baseline
        defaults are used.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Period-level simulation results and run-level summary statistics.
    """

    # Use an independent copy so the project-wide defaults cannot be modified
    # accidentally by an individual experiment.
    if threshold_cfg is None:
        threshold_cfg = DEFAULT_STABILITY_THRESHOLDS.copy()
    else:
        threshold_cfg = dict(
            threshold_cfg
        )

    validate_threshold_config(
        threshold_cfg
    )

    # -------------------------------------------------------------------------
    # Initialise the fixed-network market
    # -------------------------------------------------------------------------

    env = FixedNetworkMarketEnv(
        P_init=P,
        seed=seed,
        horizon=horizon,
    )

    rows = []

    done = False

    # -------------------------------------------------------------------------
    # Run the complete simulation path
    # -------------------------------------------------------------------------

    while not done:

        info, done = env.step()

        # Attach experiment identifiers to every period-level observation.
        info["topology"] = topology_name
        info["seed"] = int(seed)

        rows.append(info)

    df = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------------
    # Stability classification
    # -------------------------------------------------------------------------

    stable_flags = rolling_stability_flags(
        df=df,
        riskS_mean_max=threshold_cfg["riskS_mean_max"],
        ret_vol_max=threshold_cfg["ret_vol_max"],
        belief_var_max=threshold_cfg["belief_var_max"],
        flow_std_max=threshold_cfg["flow_std_max"],
        window=threshold_cfg["window"],
    )

    # Store the binary stability indicator with the raw simulation output.
    df["stable_flag"] = (
        stable_flags.astype(int)
    )

    # -------------------------------------------------------------------------
    # Time to persistent stability
    # -------------------------------------------------------------------------

    tts = time_to_stability(
        stable_flags=stable_flags,
        required_consecutive_windows=(
            threshold_cfg[
                "required_consecutive_windows"
            ]
        ),
    )

    # -------------------------------------------------------------------------
    # Run-level summary
    # -------------------------------------------------------------------------

    summary = summarize_run(
        df=df,
        topology=topology_name,
        seed=seed,
        stable_flags=stable_flags,
        time_to_stability_value=tts,
    )

    return (
        df,
        summary,
    )


# =============================================================================
# Repeated Baseline Experiment
# =============================================================================

def run_baseline_batch(
    topologies: Mapping[str, np.ndarray],
    seed_start: int,
    n_seeds: int,
    horizon: int,
    threshold_cfg: Mapping | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run repeated baseline simulations across topology classes and seeds.

    Each topology is evaluated under the same set of simulation seeds. This
    preserves a transparent experimental design and facilitates paired
    topology comparisons later in the analysis layer.

    Parameters
    ----------
    topologies : Mapping[str, np.ndarray]
        Mapping from topology name to fixed influence matrix.

    seed_start : int
        First simulation seed.

    n_seeds : int
        Number of simulation replications per topology.

    horizon : int
        Number of periods in each simulation.

    threshold_cfg : Mapping or None
        Stability-threshold configuration.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Combined period-level data and one run-level summary row per
        topology-seed combination.
    """

    if n_seeds <= 0:
        raise ValueError(
            "n_seeds must be positive."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    if not topologies:
        raise ValueError(
            "At least one topology must be supplied."
        )

    raw_frames = []
    summary_rows = []

    seed_stop = (
        seed_start
        + n_seeds
    )

    # -------------------------------------------------------------------------
    # Topology × seed experimental loop
    # -------------------------------------------------------------------------

    for topology_name, P in topologies.items():

        for seed in range(
            seed_start,
            seed_stop,
        ):

            df_run, summary = run_one_topology(
                topology_name=topology_name,
                P=P,
                seed=seed,
                horizon=horizon,
                threshold_cfg=threshold_cfg,
            )

            raw_frames.append(
                df_run
            )

            summary_rows.append(
                summary
            )

    # Combine all period-level simulations into one table.
    raw_df = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    # One row per topology × seed replication.
    summary_df = pd.DataFrame(
        summary_rows
    )

    return (
        raw_df,
        summary_df,
    )
