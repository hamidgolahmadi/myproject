"""Check that refactored baseline aggregation matches the legacy logic."""

import numpy as np
import pandas as pd

from src.analysis.aggregate_results import aggregate_baseline_by_topology


# -------------------------------------------------------------------------
# Construct deterministic run-level summaries
# -------------------------------------------------------------------------

summary_df = pd.DataFrame({
    "topology": [
        "random", "random",
        "small_world", "small_world",
        "scale_free", "scale_free",
    ],
    "riskS_mean": [1.0, 1.2, 0.9, 1.1, 1.5, 1.7],
    "peak_riskS": [2.0, 2.2, 1.8, 2.0, 2.5, 2.7],
    "belief_var_mean": [0.01, 0.02, 0.015, 0.017, 0.03, 0.04],
    "peak_belief_var": [0.03, 0.04, 0.035, 0.037, 0.05, 0.06],
    "return_vol": [0.001, 0.002, 0.0015, 0.0017, 0.0025, 0.0027],
    "cum_abs_returns": [1.0, 1.2, 0.9, 1.1, 1.5, 1.7],
    "cum_flow2": [10.0, 12.0, 9.0, 11.0, 15.0, 17.0],
    "gini_mean": [0.2, 0.22, 0.25, 0.27, 0.5, 0.55],
    "fraction_time_stable": [0.8, 0.7, 0.85, 0.82, 0.5, 0.45],
    "time_to_stability": [100.0, 120.0, 90.0, 95.0, 160.0, 170.0],
})


# -------------------------------------------------------------------------
# Legacy aggregation logic
# -------------------------------------------------------------------------

legacy = (
    summary_df.groupby("topology", as_index=False)
    .agg(
        riskS_mean=("riskS_mean", "mean"),
        peak_riskS=("peak_riskS", "mean"),
        belief_var_mean=("belief_var_mean", "mean"),
        peak_belief_var=("peak_belief_var", "mean"),
        return_vol=("return_vol", "mean"),
        cum_abs_returns=("cum_abs_returns", "mean"),
        cum_flow2=("cum_flow2", "mean"),
        gini_mean=("gini_mean", "mean"),
        fraction_time_stable=("fraction_time_stable", "mean"),
        time_to_stability=("time_to_stability", "mean"),
    )
)


# -------------------------------------------------------------------------
# Refactored aggregation
# -------------------------------------------------------------------------

refactored = aggregate_baseline_by_topology(summary_df)


# -------------------------------------------------------------------------
# Compare outputs
# -------------------------------------------------------------------------

if list(legacy.columns) != list(refactored.columns):
    raise AssertionError("Aggregation columns do not match.")

if not legacy["topology"].equals(refactored["topology"]):
    raise AssertionError("Topology ordering does not match.")

numeric_columns = [
    column
    for column in legacy.columns
    if column != "topology"
]

for column in numeric_columns:
    if not np.allclose(
        legacy[column].to_numpy(),
        refactored[column].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError(
            f"Mismatch in aggregated column: {column}"
        )


print("PASS: refactored aggregation matches legacy aggregation.")
