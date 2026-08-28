# run_interaction_topology_chunk.py

import os
import math
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from env_adaptive_credibility_v1 import InfoNetworkAdaptiveEnv


# =========================
# Fixed extra params
# =========================
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


# =========================
# Helpers
# =========================
def load_topology(topology_name):
    path = TOPOLOGY_PATHS[topology_name]
    data = np.load(path)
    return data["P"]


def get_chunk_bounds(n_total, chunk_index, n_chunks):
    chunk_size = math.ceil(n_total / n_chunks)
    start = chunk_index * chunk_size
    end = min((chunk_index + 1) * chunk_size, n_total)
    return start, end


def detect_explosion(history, price_upper=1000.0, abs_return_upper=0.25):
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


# =========================
# Core simulation
# =========================
def run_one_sim(args):
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
    avg_abs_pos = np.array([row["avg_abs_position"] for row in history], dtype=float)

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


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--chunk_index", type=int, required=True)
    parser.add_argument("--n_chunks", type=int, required=True)
    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--n_workers", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------
    # load sample grid
    # -------------------------
    df = pd.read_csv(args.samples_csv)
    start, end = get_chunk_bounds(len(df), args.chunk_index, args.n_chunks)
    df_chunk = df.iloc[start:end].copy()

    # -------------------------
    # load topology once
    # -------------------------
    P_init = load_topology(args.topology)

    param_cols = [
        "sigma_signal",
        "sigma_belief",
        "rho_y",
        "sigma_y",
        "alpha_social",
        "beta",
        "gamma",
        "trade_sensitivity",
        "price_impact",
        "sigma_price",
    ]

    out_path = os.path.join(
        args.outdir,
        f"interaction_abg_{args.topology}_chunk_{args.chunk_index:04d}.csv"
    )

    buffer = []
    buffer_size = 500

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:

        for _, srow in df_chunk.iterrows():
            sample_id = int(srow["sample_id"])

            env_params = FIXED_EXTRA_PARAMS.copy()
            for p in param_cols:
                env_params[p] = float(srow[p])

            jobs = [
                (P_init, seed, args.horizon, env_params)
                for seed in range(args.n_seeds)
            ]

            futures = {executor.submit(run_one_sim, job): job[1] for job in jobs}

            for future in as_completed(futures):
                seed = futures[future]
                result = future.result()

                row = {
                    "sample_id": sample_id,
                    "topology": args.topology,
                    "seed": seed,
                }

                for p in param_cols:
                    row[p] = float(srow[p])

                row.update(result)
                buffer.append(row)

                if len(buffer) >= buffer_size:
                    pd.DataFrame(buffer).to_csv(
                        out_path,
                        mode="a",
                        header=not os.path.exists(out_path),
                        index=False
                    )
                    buffer = []

    if buffer:
        pd.DataFrame(buffer).to_csv(
            out_path,
            mode="a",
            header=not os.path.exists(out_path),
            index=False
        )

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
