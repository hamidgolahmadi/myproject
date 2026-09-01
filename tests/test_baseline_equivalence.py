"""Check that the refactored baseline reproduces the legacy baseline."""

import importlib.util
from pathlib import Path

import numpy as np

from src.model.baseline_env import FixedNetworkMarketEnv
from src.topologies.random_network import build_P_random_fixed


# Load the legacy script even though its filename starts with a number.
legacy_path = Path("archive/legacy/02_run_baseline_no_policy_v2.py")
spec = importlib.util.spec_from_file_location("legacy_baseline", legacy_path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


# Use exactly the same topology and random seed in both implementations.
P = build_P_random_fixed(N=20, K=4, seed=123)

old_env = legacy.InfoNetworkBaselineEnv(
    P_init=P,
    seed=456,
    horizon=20,
)

new_env = FixedNetworkMarketEnv(
    P_init=P,
    seed=456,
    horizon=20,
)


# Compare the main numerical outputs period by period.
keys = [
    "price",
    "return",
    "riskS",
    "net_flow",
    "flow2",
    "belief_var",
    "signal_var",
    "position_var",
    "gini_in",
]

for t in range(20):
    old_info, old_done = old_env.step()
    new_info, new_done = new_env.step()

    for key in keys:
        if not np.isclose(
            old_info[key],
            new_info[key],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(
                f"Mismatch at t={t}, key={key}: "
                f"old={old_info[key]}, new={new_info[key]}"
            )

    assert old_done == new_done

print("PASS: refactored baseline matches legacy baseline.")
