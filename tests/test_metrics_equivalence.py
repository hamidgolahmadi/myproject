"""Check that refactored market/stability metrics match the legacy code."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.market_metrics import summarize_run
from src.metrics.stability_metrics import (
    rolling_stability_flags,
    time_to_stability,
)


# Load the legacy baseline script.
legacy_path = Path("archive/legacy/02_run_baseline_no_policy_v2.py")
spec = importlib.util.spec_from_file_location("legacy_baseline", legacy_path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


# -------------------------------------------------------------------------
# Construct deterministic synthetic period-level data
# -------------------------------------------------------------------------

rng = np.random.default_rng(123)

n = 200

df = pd.DataFrame({
    "return": rng.normal(0.0, 0.001, size=n),
    "riskS": rng.uniform(0.5, 1.5, size=n),
    "belief_var": rng.uniform(0.001, 0.010, size=n),
    "flow_std_cs": rng.uniform(0.02, 0.08, size=n),
    "abs_return": np.abs(rng.normal(0.0, 0.001, size=n)),
    "belief_range": rng.uniform(0.01, 0.20, size=n),
    "net_flow": rng.normal(0.0, 1.0, size=n),
    "flow2": rng.uniform(0.0, 1.0, size=n),
    "signal_var": rng.uniform(0.0, 0.5, size=n),
    "position_var": rng.uniform(0.0, 2.0, size=n),
    "gini_in": rng.uniform(0.0, 1.0, size=n),
})


# -------------------------------------------------------------------------
# Compare rolling stability flags
# -------------------------------------------------------------------------

old_flags = legacy.rolling_stability_flags(df)
new_flags = rolling_stability_flags(df)

if not np.array_equal(old_flags, new_flags):
    raise AssertionError("rolling_stability_flags mismatch")


# -------------------------------------------------------------------------
# Compare time to stability
# -------------------------------------------------------------------------

old_tts = legacy.time_to_stability(old_flags)
new_tts = time_to_stability(new_flags)

if not (
    (np.isnan(old_tts) and np.isnan(new_tts))
    or np.isclose(old_tts, new_tts)
):
    raise AssertionError(
        f"time_to_stability mismatch: old={old_tts}, new={new_tts}"
    )


# -------------------------------------------------------------------------
# Compare run summaries
# -------------------------------------------------------------------------

old_summary = legacy.summarize_run(
    df=df,
    topology="random_fixed",
    seed=999,
    stable_flags=old_flags,
    tts=old_tts,
)

new_summary = summarize_run(
    df=df,
    topology="random_fixed",
    seed=999,
    stable_flags=new_flags,
    time_to_stability_value=new_tts,
)


for key in old_summary:

    if key not in new_summary:
        raise AssertionError(
            f"Missing summary key in new implementation: {key}"
        )

    old_value = old_summary[key]
    new_value = new_summary[key]

    if isinstance(old_value, str):
        if old_value != new_value:
            raise AssertionError(
                f"Mismatch for {key}: old={old_value}, new={new_value}"
            )
        continue

    if (
        np.isnan(old_value)
        and np.isnan(new_value)
    ):
        continue

    if not np.isclose(
        old_value,
        new_value,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            f"Mismatch for {key}: old={old_value}, new={new_value}"
        )


print("PASS: refactored metrics match legacy metrics.")
