"""
Stability diagnostics for repeated market simulations.

This module contains metrics used to determine whether a simulated market
has entered a sufficiently calm and persistent regime.

The stability definition is intentionally operational rather than structural:
it is based on rolling empirical thresholds for market risk, return volatility,
belief dispersion, and cross-sectional order-flow dispersion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Rolling Stability Classification
# =============================================================================

def rolling_stability_flags(
    df: pd.DataFrame,
    riskS_mean_max: float = 2.0,
    ret_vol_max: float = 0.0020,
    belief_var_max: float = 0.015,
    flow_std_max: float = 0.12,
    window: int = 50,
) -> np.ndarray:
    """
    Classify each period according to rolling stability conditions.

    A period is labelled stable when all four rolling-window conditions hold:

    1. Mean scaled risk is below ``riskS_mean_max``.
    2. Return volatility is below ``ret_vol_max``.
    3. Mean belief variance is below ``belief_var_max``.
    4. Mean cross-sectional order-flow dispersion is below ``flow_std_max``.

    Parameters
    ----------
    df : pd.DataFrame
        Period-level simulation output.

    riskS_mean_max : float
        Maximum permitted rolling mean of the scaled risk metric.

    ret_vol_max : float
        Maximum permitted rolling standard deviation of returns.

    belief_var_max : float
        Maximum permitted rolling mean of belief variance.

    flow_std_max : float
        Maximum permitted rolling mean of cross-sectional flow dispersion.

    window : int
        Number of consecutive observations used in each rolling calculation.

    Returns
    -------
    np.ndarray
        Boolean array where ``True`` indicates that all rolling stability
        conditions are satisfied at that period.
    """

    n = len(df)

    # No period is classified as stable until a complete rolling window exists.
    stable = np.zeros(
        n,
        dtype=bool,
    )

    # Extract NumPy arrays once to avoid repeated DataFrame indexing inside
    # the main rolling loop.
    return_series = df["return"].to_numpy()
    risk_series = df["riskS"].to_numpy()
    belief_var_series = df["belief_var"].to_numpy()
    flow_std_series = df["flow_std_cs"].to_numpy()

    for t in range(
        window - 1,
        n,
    ):
        start = (
            t
            - window
            + 1
        )

        # Rolling average of the scaled risk proxy.
        risk_mean = float(
            np.mean(
                risk_series[start:t + 1]
            )
        )

        # Rolling return volatility.
        return_volatility = float(
            np.std(
                return_series[start:t + 1]
            )
        )

        # Rolling average belief dispersion.
        belief_mean = float(
            np.mean(
                belief_var_series[start:t + 1]
            )
        )

        # Rolling average cross-sectional order-flow dispersion.
        flow_std_mean = float(
            np.mean(
                flow_std_series[start:t + 1]
            )
        )

        # Stability requires all conditions to hold simultaneously.
        stable[t] = (
            risk_mean < riskS_mean_max
            and return_volatility < ret_vol_max
            and belief_mean < belief_var_max
            and flow_std_mean < flow_std_max
        )

    return stable


# =============================================================================
# Time to Stability
# =============================================================================

def time_to_stability(
    stable_flags: np.ndarray,
    required_consecutive_windows: int = 3,
) -> float:
    """
    Return the first period at which stability persists sufficiently long.

    A simulation is considered to have stabilised only after the stability
    condition has remained true for a specified number of consecutive periods.
    This avoids classifying a brief isolated calm period as genuine
    stabilisation.

    Parameters
    ----------
    stable_flags : np.ndarray
        Boolean stability indicator for each period.

    required_consecutive_windows : int
        Number of consecutive stable observations required before declaring
        the system stabilised.

    Returns
    -------
    float
        Index of the first period completing the required stable sequence.
        Returns ``np.nan`` if the simulation never satisfies the persistence
        criterion.
    """

    consecutive_count = 0

    for index, flag in enumerate(
        stable_flags
    ):

        if flag:
            consecutive_count += 1

            if (
                consecutive_count
                >= required_consecutive_windows
            ):
                return float(index)

        else:
            # Any unstable period breaks the current consecutive sequence.
            consecutive_count = 0

    return np.nan
