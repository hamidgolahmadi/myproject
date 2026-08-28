# -*- coding: utf-8 -*-
"""
03_run_adaptive_credibility_grid_v1.py

Paper 1 experiment runner:
- constant topology
- adaptive credibility weights
- grid over beta and gamma
- repeated over many seeds

Outputs one CSV for one chunk/job.

Recommended usage:
- run through SLURM array jobs
- each job handles one (topology, beta, gamma, seed-block) chunk
"""

import os
import argparse
import numpy as np
import pandas as pd

from env_adaptive_credibility_v1 import InfoNetworkAdaptiveCredibilityEnv


def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s < eps:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * (cum / (cum[-1] + eps)).sum()) / n
    return float(max(0.0, min(1.0, g)))


def rolling_stability_flags(
    df: pd.DataFrame,
    riskS_mean_max: float = 8.0,
    ret_vol_max: float = 0.02,
    belief_var_max: float = 0.05,
    flow_std_max: float = 0.20,
    window: int = 50,
):
    n = len(df)
    stable = np.zeros(n, dtype=bool)

    ret_series = df["return"].to_numpy()
    risk_series = df["riskS"].to_numpy()
    belief_var_series = df["belief_var"].to_numpy()
    flow_std_series = df["flow_std_cs"].to_numpy()

    for t in range(window - 1, n):
        s = t - window + 1
        risk_mean = float(np.mean(risk_series[s:t + 1]))
        ret_vol = float(np.std(ret_series[s:t + 1]))
        belief_mean = float(np.mean(belief_var_series[s:t + 1]))
        flow_std_mean = float(np.mean(flow_std_series[s:t + 1]))

        stable[t] = (
            (risk_mean < riskS_mean_max)
            and (ret_vol < ret_vol_max)
            and (belief_mean < belief_var_max)
            and (flow_std_mean < flow_std_max)
        )

    return stable


def time_to_stability(stable_flags: np.ndarray, required_consecutive_windows: int = 3) -> float:
    count = 0
    for i, flag in enumerate(stable_flags):
        if flag:
            count += 1
            if count >= required_consecutive_windows:
                return float(i)
        else:
            count = 0
    return np.nan


def load_topology(path: str) -> np.ndarray:
    data = np.load(path)
    return data["P"]


def run_one_seed(
    topology_name: str,
    P: np.ndarray,
    beta: float,
    gamma: float,
    seed: int,
    horizon: int,
    threshold_cfg: dict,
):
    env = InfoNetworkAdaptiveCredibilityEnv(
        P_init=P,
        beta=beta,
        gamma=gamma,
        seed=seed,
        horizon=horizon,
    )

    rows = []
    done = False
    while not done:
        info, done = env.step()
        rows.append(info)

    df = pd.DataFrame(rows)

    stable_flags = rolling_stability_flags(
        df,
        riskS_mean_max=threshold_cfg["riskS_mean_max"],
        ret_vol_max=threshold_cfg["ret_vol_max"],
        belief_var_max=threshold_cfg["belief_var_max"],
        flow_std_max=threshold_cfg["flow_std_max"],
        window=threshold_cfg["window"],
    )
    df["stable_flag"] = stable_flags.astype(int)

    tts = time_to_stability(
        stable_flags=stable_flags,
        required_consecutive_windows=threshold_cfg["required_consecutive_windows"],
    )

    summary = {
        "topology": topology_name,
        "beta": float(beta),
        "gamma": float(gamma),
        "seed": int(seed),
        "riskS_mean": float(df["riskS"].mean()),
        "riskS_std": float(df["riskS"].std()),
        "peak_riskS": float(df["riskS"].max()),
        "return_vol": float(df["return"].std()),
        "belief_var_mean": float(df["belief_var"].mean()),
        "peak_belief_var": float(df["belief_var"].max()),
        "flow_std_mean": float(df["flow_std_cs"].mean()),
        "gini_mean": float(df["gini_in"].mean()),
        "cum_abs_returns": float(df["abs_return"].sum()),
        "cum_flow2": float(df["flow2"].sum()),
        "fraction_time_stable": float(np.mean(stable_flags)),
        "time_to_stability": float(tts) if np.isfinite(tts) else np.nan,
        "explosive": int(float(df["riskS"].mean()) > threshold_cfg["explosion_riskS_mean"]),
    }

    return summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--topology_dir", type=str, default="extreme_topologies_v2")
    parser.add_argument("--outdir", type=str, default="adaptive_credibility_grid_v1")

    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)

    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)

    parser.add_argument("--riskS_mean_max", type=float, default=8.0)
    parser.add_argument("--ret_vol_max", type=float, default=0.02)
    parser.add_argument("--belief_var_max", type=float, default=0.05)
    parser.add_argument("--flow_std_max", type=float, default=0.20)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--required_consecutive_windows", type=int, default=3)

    parser.add_argument("--explosion_riskS_mean", type=float, default=1000.0)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    threshold_cfg = {
        "riskS_mean_max": args.riskS_mean_max,
        "ret_vol_max": args.ret_vol_max,
        "belief_var_max": args.belief_var_max,
        "flow_std_max": args.flow_std_max,
        "window": args.window,
        "required_consecutive_windows": args.required_consecutive_windows,
        "explosion_riskS_mean": args.explosion_riskS_mean,
    }

    topology_files = {
        "random_fixed_extreme": os.path.join(args.topology_dir, "random_fixed_extreme.npz"),
        "scale_free_extreme": os.path.join(args.topology_dir, "scale_free_extreme.npz"),
        "small_world_clustered": os.path.join(args.topology_dir, "small_world_clustered.npz"),
    }

    if args.topology not in topology_files:
        raise ValueError(f"Unknown topology: {args.topology}")

    topo_path = topology_files[args.topology]
    if not os.path.exists(topo_path):
        raise FileNotFoundError(f"Missing topology file: {topo_path}")

    P = load_topology(topo_path)

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        rows.append(
            run_one_seed(
                topology_name=args.topology,
                P=P,
                beta=args.beta,
                gamma=args.gamma,
                seed=seed,
                horizon=args.horizon,
                threshold_cfg=threshold_cfg,
            )
        )

    df = pd.DataFrame(rows)

    safe_topo = args.topology
    safe_beta = str(args.beta).replace(".", "p")
    safe_gamma = str(args.gamma).replace(".", "p")
    safe_seed = f"{args.seed_start}_{args.seed_start + args.n_seeds - 1}"

    outpath = os.path.join(
        args.outdir,
        f"grid_topo_{safe_topo}_beta_{safe_beta}_gamma_{safe_gamma}_seeds_{safe_seed}.csv",
    )
    df.to_csv(outpath, index=False)

    print("Saved:", outpath)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
