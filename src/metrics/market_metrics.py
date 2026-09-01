"""
Run-level market summary metrics.

This module converts period-by-period simulation output into a compact set of
run-level statistics used for topology comparison, robustness analysis, and
later statistical aggregation.

It does not define stability itself; stability flags and time-to-stability are
computed separately in ``stability_metrics.py`` and passed into this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Run-Level Market Summary
# =============================================================================

def summarize_run(
    df: pd.DataFrame,
    topology: str,
    seed: int,
    stable_flags: np.ndarray,
    time_to_stability_value: float,
) -> dict:
    """
    Summarise one complete simulation run.

    Parameters
    ----------
    df : pd.DataFrame
        Period-level simulation output.

    topology : str
        Name of the topology used in the simulation.

    seed : int
        Replication seed.

    stable_flags : np.ndarray
        Boolean stability classification for each period.

    time_to_stability_value : float
        First period satisfying the required persistent-stability criterion.
        May be ``np.nan`` when the run never stabilises.

    Returns
    -------
    dict
        Run-level market, belief, order-flow, position, and stability metrics.
    """

    return {
        # ---------------------------------------------------------------------
        # Experiment identification
        # ---------------------------------------------------------------------
        "topology": topology,
        "seed": int(seed),

        # ---------------------------------------------------------------------
        # Risk
        # ---------------------------------------------------------------------
        "riskS_mean": float(
            df["riskS"].mean()
        ),
        "riskS_std": float(
            df["riskS"].std()
        ),
        "riskS_p95": float(
            df["riskS"].quantile(0.95)
        ),
        "peak_riskS": float(
            df["riskS"].max()
        ),

        # ---------------------------------------------------------------------
        # Returns
        # ---------------------------------------------------------------------
        "return_vol": float(
            df["return"].std()
        ),
        "abs_return_mean": float(
            df["abs_return"].mean()
        ),
        "cum_abs_returns": float(
            df["abs_return"].sum()
        ),

        # ---------------------------------------------------------------------
        # Belief dispersion
        # ---------------------------------------------------------------------
        "belief_var_mean": float(
            df["belief_var"].mean()
        ),
        "belief_var_p95": float(
            df["belief_var"].quantile(0.95)
        ),
        "peak_belief_var": float(
            df["belief_var"].max()
        ),
        "belief_range_mean": float(
            df["belief_range"].mean()
        ),

        # ---------------------------------------------------------------------
        # Order flow
        # ---------------------------------------------------------------------
        "net_flow_std": float(
            df["net_flow"].std()
        ),
        "flow2_mean": float(
            df["flow2"].mean()
        ),
        "cum_flow2": float(
            df["flow2"].sum()
        ),

        # ---------------------------------------------------------------------
        # Information / signal amplification
        # ---------------------------------------------------------------------
        "signal_var_mean": float(
            df["signal_var"].mean()
        ),

        # ---------------------------------------------------------------------
        # Position dispersion
        # ---------------------------------------------------------------------
        "position_var_mean": float(
            df["position_var"].mean()
        ),

        # ---------------------------------------------------------------------
        # Network concentration
        # ---------------------------------------------------------------------
        "gini_mean": float(
            df["gini_in"].mean()
        ),

        # ---------------------------------------------------------------------
        # Stability
        # ---------------------------------------------------------------------
        "fraction_time_stable": float(
            np.mean(stable_flags)
        ),
        "time_to_stability": (
            float(time_to_stability_value)
            if np.isfinite(time_to_stability_value)
            else np.nan
        ),
    }
