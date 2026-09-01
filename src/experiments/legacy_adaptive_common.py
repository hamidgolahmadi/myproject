"""Shared legacy utilities for OAT and interaction experiments.

This module preserves the behavior of the original adaptive-credibility
experiment runners while removing duplicated simulation logic.
"""

import math

import numpy as np

from env_adaptive_credibility_v1 import InfoNetworkAdaptiveEnv


FIXED_EXTRA_PARAMS = {
    "eps_softmax": 1e-8,
    "price_floor": 1e-6,
    "lambda_risk": 0.95,
    "init_price": 100.0,
    "init_x_std": 0.1,
    "init_b_std": 0.25,
}

TOPOLOGY_PATHS = {
    "random": "extreme_topologies_v2/random_fixed_extreme.npz",
    "small_world": "extreme_topologies_v2/small_world_clustered.npz",
    "scale_free": "extreme_topologies_v2/scale_free_extreme.npz",
}


def load_topology(topology_name):
    """Load the legacy initial influence matrix for a topology."""
    path = TOPOLOGY_PATHS[topology_name]
    data = np.load(path)
    return data["P"]


def get_chunk_bounds(n_total, chunk_index, n_chunks):
    """Return the start and end indices for one parameter chunk."""
    chunk_size = math.ceil(n_total / n_chunks)
    start = chunk_index * chunk_size
    end = min((chunk_index + 1) * chunk_size, n_total)
    return start, end


def detect_explosion(history, price_upper=1000.0, abs_return_upper=0.25):
    """Apply the original legacy threshold-based explosion diagnostic."""
    prices = np.array([row["price"] for row in history], dtype=float)
    rets = np.array([row["return"] for row in history], dtype=float)

    exploded = bool(
        np.any(prices > price_upper) or np.any(np.abs(rets) > abs_return_upper)
    )

    if exploded:
        t_candidates = []
        idx_price = np.where(prices > price_upper)[0]
        idx_ret = np.where(np.abs(rets) > abs_return_upper)[0]

        if len(idx_price) > 0:
            t_candidates.append(idx_price[0])
        if len(idx_ret) > 0:
            t_candidates.append(idx_ret[0])

        t_explosion = min(t_candidates)
    else:
        t_explosion = -1

    return exploded, t_explosion


def run_one_sim(args):
    """Run one legacy adaptive simulation and return its summary statistics."""
    P_init, seed, horizon, env_params = args

    env = InfoNetworkAdaptiveEnv(
        P_init=P_init,
        seed=seed,
        horizon=horizon,
        **env_params,
    )

    history = []
    done = False

    while not done:
        row, done = env.step()
        history.append(row)

    exploded, t_explosion = detect_explosion(history)

    prices = np.array([row["price"] for row in history], dtype=float)
    rets = np.array([row["return"] for row in history], dtype=float)
    risks = np.array([row["risk_v"] for row in history], dtype=float)
    belief_vars = np.array([row["belief_var"] for row in history], dtype=float)
    avg_abs_pos = np.array(
        [row["avg_abs_position"] for row in history],
        dtype=float,
    )

    mean_return = np.mean(rets)
    mean_abs_return = np.mean(np.abs(rets))
    std_return = np.std(rets)

    return {
        "exploded": int(exploded),
        "time_to_explosion": int(t_explosion),
        "final_price": float(prices[-1]),
        "mean_return": float(mean_return),
        "mean_abs_return": float(mean_abs_return),
        "std_return": float(std_return),
        "mean_risk_v": float(np.mean(risks)),
        "mean_belief_var": float(np.mean(belief_vars)),
        "mean_avg_abs_position": float(np.mean(avg_abs_pos)),
    }
