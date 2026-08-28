# run_baseline_no_policy_v2.py
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd


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


class InfoNetworkBaselineEnv:
    def __init__(
        self,
        P_init: np.ndarray,
        seed: int = 0,
        horizon: int = 3000,
        p0: float = 100.0,
        kappa: float = 0.02,
        sigma_eps: float = 0.10,
        x_max: float = 1.0,
        tau: float = 0.001,
        rho_y: float = 0.985,
        sigma_y: float = 0.025,
        sigma_s: float = 0.06,
        omega_social: float = 0.75,
        sigma_belief: float = 0.025,
        beta_risk: float | None = None,
        risk_unit: float = 1e-6,
        gamma_fin: float = 6.0,
        gate_value: float = 0.4,
    ):
        self.P_init = np.array(P_init, dtype=float)
        self.N = self.P_init.shape[0]
        self.horizon = int(horizon)

        self.p0 = float(p0)
        self.kappa = float(kappa)
        self.sigma_eps = float(sigma_eps)
        self.x_max = float(x_max)
        self.tau = float(tau)

        self.rho_y = float(rho_y)
        self.sigma_y = float(sigma_y)
        self.sigma_s = float(sigma_s)
        self.omega_social = float(omega_social)
        self.sigma_belief = float(sigma_belief)

        self.beta_risk = float(beta_risk) if beta_risk is not None else float(2 ** (-1 / 20))
        self.risk_unit = float(risk_unit)
        self.gamma_fin = float(gamma_fin)
        self.gate_value = float(gate_value)

        self.rng = np.random.default_rng(seed)
        self.reset()

    def _private_signals(self):
        return self.y + self.rng.normal(0.0, self.sigma_s, size=self.N)

    def reset(self):
        self.t = 0
        self.P = self.P_init.copy()

        self.p = self.p0
        self.p_prev = self.p0
        self.R_prev = 0.0

        self.x = self.rng.normal(0.0, 0.1, size=self.N)
        self.c = np.zeros(self.N, dtype=float)

        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.s = self._private_signals()
        self.b = self.rng.normal(0.0, 0.25, size=self.N)

        self.risk_v = 0.0
        self.delta_x_prev = np.zeros(self.N, dtype=float)

    def step(self):
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        g_gate = np.full(self.N, self.gate_value, dtype=float)
        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * g_gate * signal)

        delta_x = a_fin * self.x_max
        tc = self.tau * np.abs(delta_x)
        self.x = self.x + delta_x

        net_flow = float(np.sum(delta_x))
        eps_p = float(self.rng.normal(0.0, self.sigma_eps))

        self.p_prev = self.p
        self.p = max(self.p + self.kappa * net_flow + eps_p, 1e-6)
        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        self.c = self.c - self.p * delta_x - tc
        self.delta_x_prev = delta_x.copy()

        indeg = self.P.sum(axis=0)
        info = {
            "t": self.t,
            "price": float(self.p),
            "return": float(R),
            "abs_return": float(abs(R)),
            "riskS": float(self.risk_v / max(self.risk_unit, 1e-18)),
            "net_flow": float(net_flow),
            "flow2": float(np.mean(delta_x ** 2)),
            "flow_std_cs": float(np.std(delta_x)),
            "belief_var": float(np.var(self.b)),
            "belief_range": float(np.max(self.b) - np.min(self.b)),
            "signal_var": float(np.var(signal)),
            "mean_abs_signal": float(np.mean(np.abs(signal))),
            "mean_abs_deltax": float(np.mean(np.abs(delta_x))),
            "position_var": float(np.var(self.x)),
            "gini_in": float(gini_coefficient(indeg)),
            "dP": 0.0,
        }

        self.t += 1
        done = self.t >= self.horizon
        return info, done


def rolling_stability_flags(
    df: pd.DataFrame,
    riskS_mean_max: float = 2.0,
    ret_vol_max: float = 0.0020,
    belief_var_max: float = 0.015,
    flow_std_max: float = 0.12,
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

        cond = (
            (risk_mean < riskS_mean_max)
            and (ret_vol < ret_vol_max)
            and (belief_mean < belief_var_max)
            and (flow_std_mean < flow_std_max)
        )
        stable[t] = cond

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


def summarize_run(df: pd.DataFrame, topology: str, seed: int, stable_flags: np.ndarray, tts: float) -> dict:
    return {
        "topology": topology,
        "seed": seed,
        "riskS_mean": float(df["riskS"].mean()),
        "riskS_std": float(df["riskS"].std()),
        "riskS_p95": float(df["riskS"].quantile(0.95)),
        "peak_riskS": float(df["riskS"].max()),
        "return_vol": float(df["return"].std()),
        "abs_return_mean": float(df["abs_return"].mean()),
        "belief_var_mean": float(df["belief_var"].mean()),
        "belief_var_p95": float(df["belief_var"].quantile(0.95)),
        "peak_belief_var": float(df["belief_var"].max()),
        "belief_range_mean": float(df["belief_range"].mean()),
        "net_flow_std": float(df["net_flow"].std()),
        "flow2_mean": float(df["flow2"].mean()),
        "signal_var_mean": float(df["signal_var"].mean()),
        "position_var_mean": float(df["position_var"].mean()),
        "gini_mean": float(df["gini_in"].mean()),
        "cum_abs_returns": float(df["abs_return"].sum()),
        "cum_flow2": float(df["flow2"].sum()),
        "fraction_time_stable": float(np.mean(stable_flags)),
        "time_to_stability": float(tts) if np.isfinite(tts) else np.nan,
    }


def load_topology(path: str) -> np.ndarray:
    data = np.load(path)
    return data["P"]


def run_one_topology(topology_name: str, P: np.ndarray, seed: int, horizon: int, threshold_cfg: dict):
    env = InfoNetworkBaselineEnv(P_init=P, seed=seed, horizon=horizon)
    rows = []

    done = False
    while not done:
        info, done = env.step()
        info["topology"] = topology_name
        info["seed"] = seed
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
    summary = summarize_run(df, topology_name, seed, stable_flags, tts)
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology_dir", type=str, default="topologies_extreme")
    parser.add_argument("--outdir", type=str, default="baseline_no_policy_v2")
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=10)

    parser.add_argument("--riskS_mean_max", type=float, default=2.0)
    parser.add_argument("--ret_vol_max", type=float, default=0.0020)
    parser.add_argument("--belief_var_max", type=float, default=0.015)
    parser.add_argument("--flow_std_max", type=float, default=0.12)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--required_consecutive_windows", type=int, default=3)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    threshold_cfg = {
        "riskS_mean_max": args.riskS_mean_max,
        "ret_vol_max": args.ret_vol_max,
        "belief_var_max": args.belief_var_max,
        "flow_std_max": args.flow_std_max,
        "window": args.window,
        "required_consecutive_windows": args.required_consecutive_windows,
    }

    topology_files = {
        "random_fixed_extreme": os.path.join(args.topology_dir, "random_fixed_extreme.npz"),
        "scale_free_extreme": os.path.join(args.topology_dir, "scale_free_extreme.npz"),
        "small_world_clustered": os.path.join(args.topology_dir, "small_world_clustered.npz"),
    }

    raw_frames = []
    summary_rows = []

    for topo_name, topo_path in topology_files.items():
        if not os.path.exists(topo_path):
            raise FileNotFoundError(f"Missing topology file: {topo_path}")

        P = load_topology(topo_path)

        for seed in range(args.seed_start, args.seed_start + args.n_seeds):
            df_run, summary = run_one_topology(
                topology_name=topo_name,
                P=P,
                seed=seed,
                horizon=args.horizon,
                threshold_cfg=threshold_cfg,
            )
            raw_frames.append(df_run)
            summary_rows.append(summary)

    raw_df = pd.concat(raw_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    agg_df = (
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

    raw_df.to_csv(os.path.join(args.outdir, "baseline_raw_v2.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "baseline_summary_by_seed_v2.csv"), index=False)
    agg_df.to_csv(os.path.join(args.outdir, "baseline_summary_v2.csv"), index=False)

    threshold_text = pd.DataFrame([threshold_cfg])
    threshold_text.to_csv(os.path.join(args.outdir, "stability_threshold_v2.csv"), index=False)

    print("Saved:")
    print(os.path.join(args.outdir, "baseline_raw_v2.csv"))
    print(os.path.join(args.outdir, "baseline_summary_by_seed_v2.csv"))
    print(os.path.join(args.outdir, "baseline_summary_v2.csv"))
    print(os.path.join(args.outdir, "stability_threshold_v2.csv"))
    print()
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
