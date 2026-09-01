# -*- coding: utf-8 -*-
"""
02_run_baseline_no_policy.py

Run market/belief dynamics on pre-generated topologies with NO policy intervention.

Outputs:
- results/baseline_raw.csv
- results/baseline_summary.csv

Usage:
    python 02_run_baseline_no_policy.py --topologies_dir topologies --horizon 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# Utilities
# =============================================================================

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


# =============================================================================
# Baseline simulator
# =============================================================================

class BaselineNoPolicySimulator:
    def __init__(
        self,
        p: np.ndarray,
        seed: int = 0,
        horizon: int = 400,
        p0: float = 100.0,
        kappa: float = 0.02,
        sigma_eps: float = 0.10,
        x_max: float = 1.0,
        rho_y: float = 0.98,
        sigma_y: float = 0.02,
        sigma_s: float = 0.05,
        omega_social: float = 0.7,
        sigma_belief: float = 0.02,
        beta_risk: float | None = None,
        risk_unit: float = 1e-6,
    ):
        self.P = p.copy()
        self.N = p.shape[0]
        self.horizon = int(horizon)

        self.p0 = float(p0)
        self.kappa = float(kappa)
        self.sigma_eps = float(sigma_eps)
        self.x_max = float(x_max)

        self.rho_y = float(rho_y)
        self.sigma_y = float(sigma_y)
        self.sigma_s = float(sigma_s)
        self.omega_social = float(omega_social)
        self.sigma_belief = float(sigma_belief)

        self.beta_risk = float(beta_risk) if beta_risk is not None else float(2 ** (-1 / 20))
        self.risk_unit = float(risk_unit)

        self.rng = np.random.default_rng(seed)
        self.gamma_fin = 5.0
        self.gate_value = 0.4

        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.price = self.p0
        self.price_prev = self.p0
        self.R_prev = 0.0

        self.x = self.rng.normal(0.0, 0.1, size=self.N)
        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.b = self.rng.normal(0.0, 0.2, size=self.N)
        self.risk_v = 0.0

    def _private_signals(self) -> np.ndarray:
        return self.y + self.rng.normal(0.0, self.sigma_s, size=self.N)

    def step(self) -> Dict[str, float]:
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        s = self._private_signals()

        pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * s + self.omega_social * pb_old + eta_b

        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * self.gate_value * signal)

        delta_x = a_fin * self.x_max
        self.x = self.x + delta_x

        net_flow = float(delta_x.sum())
        eps_p = float(self.rng.normal(0.0, self.sigma_eps))

        self.price_prev = self.price
        self.price = max(1e-6, self.price + self.kappa * net_flow + eps_p)
        r = (self.price - self.price_prev) / self.price_prev
        self.R_prev = float(r)

        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (r * r)

        indeg = self.P.sum(axis=0)

        row = {
            "t": self.t,
            "riskS": float(self.risk_v / max(self.risk_unit, 1e-18)),
            "belief_var": float(np.var(self.b)),
            "gini": float(gini_coefficient(indeg)),
            "dP": 0.0,
            "flow2": float(np.mean(delta_x ** 2)),
            "ret": float(r),
            "abs_ret": float(abs(r)),
            "price": float(self.price),
        }

        self.t += 1
        return row

    def run(self) -> pd.DataFrame:
        rows = []
        for _ in range(self.horizon):
            rows.append(self.step())
        return pd.DataFrame(rows)


# =============================================================================
# Stability rule
# =============================================================================

def first_stability_hit(
    df: pd.DataFrame,
    window: int = 30,
    risk_thresh: float = 3.0,
    belief_var_thresh: float = 0.020,
    abs_ret_thresh: float = 0.0030,
) -> int | None:
    for t in range(window - 1, len(df)):
        sub = df.iloc[t - window + 1 : t + 1]
        if (
            sub["riskS"].mean() <= risk_thresh
            and sub["belief_var"].mean() <= belief_var_thresh
            and sub["abs_ret"].mean() <= abs_ret_thresh
        ):
            return int(t)
    return None


# =============================================================================
# Main
# =============================================================================

def parse_topology_name_seed(path: Path) -> tuple[str, int]:
    stem = path.stem
    seed = int(stem.split("_seed_")[-1])
    topo = stem.split("_seed_")[0]
    return topo, seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies_dir", type=str, default="topologies")
    parser.add_argument("--out_results", type=str, default="results")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    Path(args.out_results).mkdir(parents=True, exist_ok=True)

    raw_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, float]] = []

    files = sorted(Path(args.topologies_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError("No topology files found. Run 01_generate_extreme_topologies.py first.")

    for f in files:
        topo, seed = parse_topology_name_seed(f)
        p = np.load(f)["P"]

        sim = BaselineNoPolicySimulator(p=p, seed=seed, horizon=args.horizon)
        df = sim.run()
        stab_t = first_stability_hit(df, window=args.window)

        df["topology"] = topo
        df["seed"] = seed
        raw_rows.append(df)

        summary_rows.append({
            "topology": topo,
            "seed": seed,
            "riskS_mean": float(df["riskS"].mean()),
            "belief_var_mean": float(df["belief_var"].mean()),
            "gini_mean": float(df["gini"].mean()),
            "abs_ret_mean": float(df["abs_ret"].mean()),
            "stable_hit": int(stab_t is not None),
            "time_to_stability": -1 if stab_t is None else int(stab_t),
        })

    raw_df = pd.concat(raw_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    raw_df.to_csv(Path(args.out_results) / "baseline_raw.csv", index=False)
    summary_df.to_csv(Path(args.out_results) / "baseline_summary.csv", index=False)

    print("Saved baseline_raw.csv and baseline_summary.csv")
    print(summary_df.groupby("topology")[["riskS_mean", "belief_var_mean", "abs_ret_mean", "stable_hit", "time_to_stability"]].mean())


if __name__ == "__main__":
    main()
