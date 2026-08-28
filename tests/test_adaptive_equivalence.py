"""Check that the refactored adaptive model reproduces the legacy model."""

import importlib.util
from pathlib import Path

import numpy as np

from src.model.adaptive_env import AdaptiveCredibilityMarketEnv
from src.topologies.random_network import build_P_random_fixed


# Load the legacy adaptive environment directly from its original file.
legacy_path = Path("env_adaptive_credibility_v1.py")
spec = importlib.util.spec_from_file_location("legacy_adaptive", legacy_path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


# Use exactly the same topology and simulation seed.
P = build_P_random_fixed(
    N=20,
    K=4,
    seed=123,
)

old_env = legacy.InfoNetworkAdaptiveEnv(
    P_init=P,
    seed=456,
    horizon=20,
)

new_env = AdaptiveCredibilityMarketEnv(
    P_init=P,
    seed=456,
    horizon=20,
)


# Compare the main period-level outputs.
keys = [
    "price",
    "return",
    "risk_v",
    "belief_var",
    "avg_reputation",
    "avg_abs_position",
    "net_flow",
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

    # Check internal adaptive states as well.
    if not np.allclose(
        old_env.R,
        new_env.R,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            f"Reputation mismatch at t={t}"
        )

    if not np.allclose(
        old_env.b,
        new_env.b,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            f"Belief mismatch at t={t}"
        )

    assert old_done == new_done


print("PASS: refactored adaptive model matches legacy adaptive model.")
