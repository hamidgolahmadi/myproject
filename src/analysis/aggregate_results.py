"""
Aggregation utilities for repeated simulation results.

This module converts run-level simulation summaries into topology-level
statistics across repeated seeds.

It is intentionally separate from the experiment runner:

    experiments/
        generates simulation outcomes.

    analysis/
        aggregates and compares those outcomes.

The default aggregation below reproduces the topology-level summary used in
the legacy baseline script.
"""

from __future__ import annotations

import pandas as pd


# =============================================================================
# Baseline Topology Aggregation
# =============================================================================

def aggregate_baseline_by_topology(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate run-level baseline results across seeds for each topology.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Run-level summary table containing one row per topology-seed
        simulation.

    Returns
    -------
    pd.DataFrame
        Topology-level averages of the main market, belief, network, and
        stability metrics.

    Raises
    ------
    ValueError
        If the input DataFrame is empty.

    KeyError
        If required columns are missing.
    """

    if summary_df.empty:
        raise ValueError(
            "summary_df is empty; no results are available to aggregate."
        )

    required_columns = {
        "topology",
        "riskS_mean",
        "peak_riskS",
        "belief_var_mean",
        "peak_belief_var",
        "return_vol",
        "cum_abs_returns",
        "cum_flow2",
        "gini_mean",
        "fraction_time_stable",
        "time_to_stability",
    }

    missing_columns = (
        required_columns
        - set(summary_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required aggregation columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # Aggregate the run-level outcomes by topology.
    #
    # This reproduces the legacy baseline summary exactly: each statistic is
    # averaged across simulation seeds within each topology class.
    aggregated = (
        summary_df
        .groupby(
            "topology",
            as_index=False,
        )
        .agg(
            riskS_mean=(
                "riskS_mean",
                "mean",
            ),
            peak_riskS=(
                "peak_riskS",
                "mean",
            ),
            belief_var_mean=(
                "belief_var_mean",
                "mean",
            ),
            peak_belief_var=(
                "peak_belief_var",
                "mean",
            ),
            return_vol=(
                "return_vol",
                "mean",
            ),
            cum_abs_returns=(
                "cum_abs_returns",
                "mean",
            ),
            cum_flow2=(
                "cum_flow2",
                "mean",
            ),
            gini_mean=(
                "gini_mean",
                "mean",
            ),
            fraction_time_stable=(
                "fraction_time_stable",
                "mean",
            ),
            time_to_stability=(
                "time_to_stability",
                "mean",
            ),
        )
    )

    return aggregated
