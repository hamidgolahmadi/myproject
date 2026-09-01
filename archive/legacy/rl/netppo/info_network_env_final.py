# -*- coding: utf-8 -*-
"""
info_network_env_final.py

Finalized environment with injectable baseline topologies:
- random_fixed
- scale_free
- small_world

Key fix (from v10.8):
- P_prev is updated EVERY step as previous-step P
  -> if no rewiring, dP should stay ~0

Compatible with your PPO scripts (obs_reg / obs_net + env.step(...)->info including dP, b, R, etc.)
"""

from __future__ import annotations

import numpy as np


# -------------------------
# Helpers
# -------------------------

def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    s = x.sum()
    if s < eps:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * (cum / (cum[-1] + eps)).sum()) / n
    return float(max(0.0, min(1.0, g)))

def row_stochastic_from_neighbors(N: int, neighbors: np.ndarray, w: np.ndarray) -> np.ndarray:
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        P[i, neighbors[i]] = w[i]
    return P

def clip_value(x: float, cap: float) -> float:
    if cap is None or cap <= 0:
        return x
    return float(np.clip(x, -cap, cap))

def _validate_P(P: np.ndarray, tol: float = 1e-6) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square (N,N)")
    if np.any(P < -1e-12):
        raise ValueError("P has negative entries")
    rs = P.sum(axis=1)
    if not np.allclose(rs, 1.0, atol=tol):
        raise ValueError(f"P rows must sum to 1.0 (row sums range: {rs.min()}..{rs.max()})")
    # not enforcing diag=0 (some topologies might allow), but your builders use diag=0 implicitly.


def build_P_from_topology(
    N: int,
    K: int,
    topology: str,
    beta: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    Uses your baseline builders.
    Expected modules in same folder:
      - random_fixed_network.py (build_P_random_fixed)
      - scale_free_network.py   (build_P_scale_free)
      - small_world_network.py  (build_P_small_world)
    """
    topo = str(topology).lower().strip()
    if topo in ("random", "random_fixed", "random-fixed"):
        from random_fixed_network import build_P_random_fixed
        P = build_P_random_fixed(N=N, K=K, seed=seed)

    elif topo in ("scale_free", "scalefree", "scale-free"):
        from scale_free_network import build_P_scale_free
        P = build_P_scale_free(N=N, K=K, seed=seed)

    elif topo in ("small_world", "smallworld", "small-world"):
        from small_world_network import build_P_small_world
        P = build_P_small_world(N=N, K=K, beta=float(beta), seed=seed)

    else:
        raise ValueError(f"Unknown topology='{topology}'. Use: random_fixed | scale_free | small_world")

    _validate_P(P)
    return P


# -------------------------
# Environment
# -------------------------

class InfoNetworkBondEnvFinal:
    """
    Mechanics:
      - Beliefs: private signal + DeGroot diffusion via P
      - Price: impact from aggregate flow + noise
      - Rewiring: applied every net_period steps (only if neighbors,w provided)

    Observations:
      - obs_reg: LIMITED (public-ish)
      - obs_net: RICH (platform)

    Reward inside env:
      - r_reg (stability shaping): risk/flow/gini/dP (NO info-quality here)

    IMPORTANT FIX:
      - P_prev is updated each step to previous-step P (even when no rewiring)
    """

    def __init__(
        self,
        # topology / baseline
        topology: str = "random_fixed",
        beta_small_world: float = 0.1,
        topology_seed: int = 0,

        # core sizes
        N: int = 50,
        K: int = 5,
        net_period: int = 5,
        indeg_cap: int = 8,          # kept for compatibility; env doesn't enforce it
        horizon: int = 1000,

        # price / trading
        p0: float = 100.0,
        kappa: float = 0.02,
        sigma_eps: float = 0.1,
        x_max: float = 1.0,
        tau: float = 0.001,

        # latent process / beliefs
        rho_y: float = 0.98,
        sigma_y: float = 0.02,
        sigma_s: float = 0.05,
        omega_social: float = 0.7,
        sigma_belief: float = 0.02,
        beta_risk: float | None = None,

        # regulator shaping weights
        risk_unit: float = 1e-6,
        w_risk: float = 1.0,
        w_flow2: float = 0.25,
        w_flow4: float = 1.0,
        w_gini: float = 0.5,
        w_net: float = 0.05,

        cap_riskS: float = 50.0,
        cap_flow4: float = 5.0,
        cap_reward: float = 50.0,

        # rng seed for the process (NOT topology seed)
        seed: int = 0,
    ):
        self.topology = topology
        self.beta_small_world = float(beta_small_world)
        self.topology_seed = int(topology_seed)

        self.N = int(N)
        self.K = int(K)
        self.net_period = int(net_period)
        self.indeg_cap = int(indeg_cap)
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
        self.w_risk = float(w_risk)
        self.w_flow2 = float(w_flow2)
        self.w_flow4 = float(w_flow4)
        self.w_gini = float(w_gini)
        self.w_net = float(w_net)

        self.cap_riskS = cap_riskS
        self.cap_flow4 = cap_flow4
        self.cap_reward = cap_reward

        self.rng = np.random.default_rng(int(seed))

        # runtime-set by trainer
        self.gamma_fin = 5.0

        self.reset()

    def _private_signals(self) -> np.ndarray:
        noise = self.rng.normal(0.0, self.sigma_s, size=self.N)
        return self.y + noise

    def _build_obs_reg(self):
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        vol20 = float(np.sqrt(max(self.risk_v, 0.0)))
        mean_abs_deltax = float(np.mean(np.abs(self.delta_x_prev)))

        obs = []
        for i in range(self.N):
            obs.append(np.array([
                self.p,
                self.R_prev,
                vol20,
                mean_abs_deltax,
                gini_in,
                in_deg[i],
            ], dtype=float))
        return obs

    def _build_obs_net(self):
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        Pb = self.P @ self.b
        var_b = float(np.var(self.b))
        vol20 = float(np.sqrt(max(self.risk_v, 0.0)))
        mean_abs_deltax = float(np.mean(np.abs(self.delta_x_prev)))

        obs = []
        for i in range(self.N):
            obs.append(np.array([
                self.p,
                self.x[i],
                self.c[i],
                self.b[i],
                self.s[i],
                Pb[i],
                in_deg[i],
                var_b,
                self.R_prev,
                vol20,
                mean_abs_deltax,
                gini_in,
            ], dtype=float))
        return obs

    def reset(self):
        self.t = 0
        self.p = self.p0
        self.p_prev = self.p0
        self.R_prev = 0.0

        self.x = self.rng.normal(0.0, 0.1, size=self.N)
        self.c = np.zeros(self.N, dtype=float)

        # Baseline topology injected here
        self.P = build_P_from_topology(
            N=self.N,
            K=self.K,
            topology=self.topology,
            beta=self.beta_small_world,
            seed=self.topology_seed,
        )
        self.P_prev = self.P.copy()

        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.s = self._private_signals()
        self.b = self.rng.normal(0.0, 0.2, size=self.N)

        self.risk_v = 0.0
        self.delta_x_prev = np.zeros(self.N, dtype=float)

        return self._build_obs_reg(), self._build_obs_net()

    def step(self, g_gate, neighbors=None, w=None):
        # IMPORTANT: P_prev should always reflect previous-step P
        P_old_step = self.P.copy()

        # Apply rewiring only if provided and on net steps
        if (self.t % self.net_period) == 0 and neighbors is not None and w is not None:
            self.P = row_stochastic_from_neighbors(self.N, neighbors, w)

        self.P_prev = P_old_step

        # latent
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        # belief update
        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        # finance action (gate dampens reaction)
        g_gate = np.clip(np.asarray(g_gate, dtype=float), 0.0, 1.0)
        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * g_gate * signal)

        delta_x = a_fin * self.x_max
        tc = self.tau * np.abs(delta_x)

        self.x = self.x + delta_x

        # price
        net_flow = float(np.sum(delta_x))
        eps_p = float(self.rng.normal(0.0, self.sigma_eps))

        self.p_prev = self.p
        self.p = self.p + self.kappa * net_flow + eps_p
        self.p = max(self.p, 1e-6)

        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        # risk EWMA
        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        # cash
        self.c = self.c - self.p * delta_x - tc
        self.delta_x_prev = delta_x.copy()

        # regulator reward
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        dP = float(np.mean(np.abs(self.P - self.P_prev)))

        flow2 = float(np.mean(delta_x ** 2))
        flow4 = float(np.mean(delta_x ** 4))

        riskS = float(self.risk_v / max(self.risk_unit, 1e-18))
        riskS_c = clip_value(riskS, self.cap_riskS)
        flow4_c = float(np.clip(flow4, 0.0, self.cap_flow4)) if self.cap_flow4 else flow4

        r_reg = 0.0
        r_reg -= self.w_risk * riskS_c
        r_reg -= self.w_flow2 * flow2
        r_reg -= self.w_flow4 * flow4_c
        r_reg -= self.w_gini * gini_in
        r_reg -= self.w_net * dP

        if self.cap_reward is not None and self.cap_reward > 0:
            r_reg = float(np.clip(r_reg, -self.cap_reward, self.cap_reward))

        self.t += 1
        done = self.t >= self.horizon

        info = {
            "riskS": float(riskS),
            "flow2": float(flow2),
            "flow4": float(flow4),
            "gini": float(gini_in),
            "dP": float(dP),
            "R": float(R),
            "price": float(self.p),
            "b": self.b.copy(),
            "signal": signal.copy(),
            "g_mean": float(np.mean(g_gate)),
        }
        return (self._build_obs_reg(), self._build_obs_net()), float(r_reg), done, info
