"""Check that the refactored baseline experiment matches the legacy runner."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.baseline import run_one_topology
from src.topologies.random_network import build_P_random_fixed


# -------------------------------------------------------------------------
# Load legacy baseline script
# -------------------------------------------------------------------------

legacy_path = Path("02_run_baseline_no_policy_v2.py")

spec = importlib.util.spec_from_file_location(
    "legacy_baseline",
    legacy_path,
)

legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


# -------------------------------------------------------------------------
# Common experimental setup
# -------------------------------------------------------------------------

P = build_P_random_fixed(
    N=20,
    K=4,
    seed=123,
)

threshold_cfg = {
    "riskS_mean_max": 2.0,
    "ret_vol_max": 0.0020,
    "belief_var_max": 0.015,
    "flow_std_max": 0.12,
    "window": 10,
    "required_consecutive_windows": 3,
}


# -------------------------------------------------------------------------
# Run legacy experiment
# -------------------------------------------------------------------------

old_df, old_summary = legacy.run_one_topology(
    topology_name="random_fixed",
    P=P,
    seed=456,
    horizon=50,
    threshold_cfg=threshold_cfg,
)


# -------------------------------------------------------------------------
# Run refactored experiment
# -------------------------------------------------------------------------

new_df, new_summary = run_one_topology(
    topology_name="random_fixed",
    P=P,
    seed=456,
    horizon=50,
    threshold_cfg=threshold_cfg,
)


# -------------------------------------------------------------------------
# Compare period-level outputs
# -------------------------------------------------------------------------

if list(old_df.columns) != list(new_df.columns):
    raise AssertionError(
        f"Column mismatch:\n"
        f"old={list(old_df.columns)}\n"
        f"new={list(new_df.columns)}"
    )


for column in old_df.columns:

    if old_df[column].dtype == object:

        if not old_df[column].equals(
            new_df[column]
        ):
            raise AssertionError(
                f"Mismatch in column: {column}"
            )

    else:

        if not np.allclose(
            old_df[column].to_numpy(),
            new_df[column].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise AssertionError(
                f"Numerical mismatch in column: {column}"
            )


# -------------------------------------------------------------------------
# Compare run-level summary
# -------------------------------------------------------------------------

if set(old_summary) != set(new_summary):
    raise AssertionError(
        "Summary keys do not match."
    )


for key in old_summary:

    old_value = old_summary[key]
    new_value = new_summary[key]

    if isinstance(old_value, str):

        if old_value != new_value:
            raise AssertionError(
                f"Mismatch for {key}: "
                f"old={old_value}, new={new_value}"
            )

    else:

        if not np.isclose(
            old_value,
            new_value,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise AssertionError(
                f"Mismatch for {key}: "
                f"old={old_value}, new={new_value}"
            )


print("PASS: refactored baseline experiment matches legacy runner.")
