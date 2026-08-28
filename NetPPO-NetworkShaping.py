# train_netppo_causalprobe_v10_11.py
# -*- coding: utf-8 -*-
"""
NetPPO-CausalProbe v10.11 (unified, single-file)

Patched defaults:
- net_temperature default = 1.0
- net_ent_coef default = 0.0
- rewiring does NOT start at t=0 (net_start_t default = net_period)
- dP is computed as mean(|P_after - P_before|) using P_before vs P_after

Optional probe reward (recommended for sanity-checking learnability):
r_probe = -(lam_risk * clip(riskS) + lam_gini * gini + lam_dP * dP)
Use: --use_probe_reward 1

Examples:
  python train_netppo_causalprobe_v10_11.py --topology random_fixed --seed 0
  python train_netppo_causalprobe_v10_11.py --topology scale_free   --seed 0
  python train_netppo_causalprobe_v10_11.py --topology small_world  --beta 0.1 --seed 0
"""

import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Utils
# =============================================================================

LOG_2PI = float(math.log(2.0 * math.pi))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s < eps:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * (cum / (cum[-1] + eps)).sum()) / n
    return float(max(0.0, min(1.0, g)))


# =============================================================================
# Topology builders (row-stochastic P)
# =============================================================================

def _row_stochastic_from_neighbors(N: int, neighbors: np.ndarray, w: np.ndarray) -> np.ndarray:
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        P[i, neighbors[i]] = w[i]
    return P


def build_random_fixed_network(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        candidates = [j for j in range(N) if j != i]
        nbrs = rng.choice(candidates, size=K, replace=False)
        w = rng.random(K)
        w = w / w.sum()
        P[i, nbrs] = w
    return P


def build_scale_free_network(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    indeg = np.ones(N, dtype=float)
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        probs = indeg / indeg.sum()
        probs = probs.copy()
        probs[i] = 0.0
        probs = probs / probs.sum()

        nbrs = rng.choice(np.arange(N), size=K, replace=False, p=probs)
        w = rng.random(K)
        w = w / w.sum()
        P[i, nbrs] = w
        indeg[nbrs] += 1.0
    return P


def build_small_world_network(N: int, K: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    P = np.zeros((N, N), dtype=float)

    base_neighbors: List[List[int]] = []
    for i in range(N):
        nbrs = [((i + 1 + k) % N) for k in range(K)]
        base_neighbors.append(nbrs)

    for i in range(N):
        nbrs = base_neighbors[i].copy()
        for kk in range(K):
            if rng.random() < beta:
                forbidden = set(nbrs)
                forbidden.add(i)
                candidates = [j for j in range(N) if j not in forbidden]
                if candidates:
                    nbrs[kk] = int(rng.choice(candidates))

        w = rng.random(K)
        w = w / w.sum()
        P[i, np.asarray(nbrs, dtype=int)] = w

    return P


# =============================================================================
# Environment
# =============================================================================

class InfoNetworkBondEnvFinal:
    def __init__(
        self,
        topology: str = "random_fixed",
        beta_small_world: float = 0.1,
        topology_seed: int = 0,
        seed: int = 0,
        N: int = 50,
        K: int = 5,
        net_period: int = 5,
        indeg_cap: int = 8,
        horizon: int = 1000,
        p0: float = 100.0,
        kappa: float = 0.02,
        sigma_eps: float = 0.1,
        x_max: float = 1.0,
        tau: float = 0.001,
        rho_y: float = 0.98,
        sigma_y: float = 0.02,
        sigma_s: float = 0.05,
        omega_social: float = 0.7,
        sigma_belief: float = 0.02,
        beta_risk: Optional[float] = None,
        risk_unit: float = 1e-6,
        w_risk: float = 1.0,
        w_flow2: float = 0.25,
        w_flow4: float = 1.0,
        w_gini: float = 0.5,
        w_net: float = 0.05,
        cap_riskS: float = 50.0,
        cap_flow4: float = 5.0,
        cap_reward: float = 50.0,
    ):
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
        self.cap_riskS = float(cap_riskS)
        self.cap_flow4 = float(cap_flow4)
        self.cap_reward = float(cap_reward)

        self.rng = np.random.default_rng(seed)
        self.rng_topo = np.random.default_rng(topology_seed)

        self.topology = str(topology)
        self.beta_small_world = float(beta_small_world)

        self.gamma_fin = 5.0

        self.P0 = self._init_topology_P()
        self.reset()

    def _init_topology_P(self) -> np.ndarray:
        if self.topology == "random_fixed":
            return build_random_fixed_network(self.N, self.K, self.rng_topo)
        if self.topology == "scale_free":
            return build_scale_free_network(self.N, self.K, self.rng_topo)
        if self.topology == "small_world":
            return build_small_world_network(self.N, self.K, self.beta_small_world, self.rng_topo)
        raise ValueError(f"Unknown topology: {self.topology}")

    def _private_signals(self) -> np.ndarray:
        noise = self.rng.normal(0.0, self.sigma_s, size=self.N)
        return self.y + noise

    def _build_obs_reg(self) -> List[np.ndarray]:
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        vol20 = float(np.sqrt(max(self.risk_v, 0.0)))
        mean_abs_deltax = float(np.mean(np.abs(self.delta_x_prev)))

        obs: List[np.ndarray] = []
        for i in range(self.N):
            obs.append(
                np.array(
                    [self.p, self.R_prev, vol20, mean_abs_deltax, gini_in, in_deg[i]],
                    dtype=float,
                )
            )
        return obs

    def _build_obs_net(self) -> List[np.ndarray]:
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        Pb = self.P @ self.b
        var_b = float(np.var(self.b))
        vol20 = float(np.sqrt(max(self.risk_v, 0.0)))
        mean_abs_deltax = float(np.mean(np.abs(self.delta_x_prev)))

        obs: List[np.ndarray] = []
        for i in range(self.N):
            obs.append(
                np.array(
                    [
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
                    ],
                    dtype=float,
                )
            )
        return obs

    def reset(self):
        self.t = 0
        self.p = self.p0
        self.p_prev = self.p0
        self.R_prev = 0.0

        self.x = self.rng.normal(0.0, 0.1, size=self.N)
        self.c = np.zeros(self.N, dtype=float)

        self.P = self.P0.copy()

        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.s = self._private_signals()
        self.b = self.rng.normal(0.0, 0.2, size=self.N)

        self.risk_v = 0.0
        self.delta_x_prev = np.zeros(self.N, dtype=float)

        return self._build_obs_reg(), self._build_obs_net()

    def step(self, g_gate, neighbors=None, w=None):
        dP = 0.0
        if (self.t % self.net_period) == 0 and (neighbors is not None) and (w is not None):
            P_before = self.P.copy()
            self.P = _row_stochastic_from_neighbors(self.N, neighbors, w)
            dP = float(np.mean(np.abs(self.P - P_before)))

        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        g_gate = np.clip(np.asarray(g_gate, dtype=float), 0.0, 1.0)
        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * g_gate * signal)

        delta_x = a_fin * self.x_max
        tc = self.tau * np.abs(delta_x)
        self.x = self.x + delta_x

        net_flow = float(np.sum(delta_x))
        eps_p = float(self.rng.normal(0.0, self.sigma_eps))
        self.p_prev = self.p
        self.p = self.p + self.kappa * net_flow + eps_p
        self.p = max(self.p, 1e-6)

        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        self.c = self.c - self.p * delta_x - tc
        self.delta_x_prev = delta_x.copy()

        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)

        flow2 = float(np.mean(delta_x ** 2))
        flow4 = float(np.mean(delta_x ** 4))

        riskS = float(self.risk_v / max(self.risk_unit, 1e-18))
        riskS_c = float(np.clip(riskS, -self.cap_riskS, self.cap_riskS))
        flow4_c = float(np.clip(flow4, 0.0, self.cap_flow4)) if self.cap_flow4 else flow4

        r_reg = 0.0
        r_reg -= self.w_risk * riskS_c
        r_reg -= self.w_flow2 * flow2
        r_reg -= self.w_flow4 * flow4_c
        r_reg -= self.w_gini * gini_in
        r_reg -= self.w_net * dP

        if self.cap_reward and self.cap_reward > 0:
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


# =============================================================================
# Net sampling (indegree cap)
# =============================================================================

@torch.no_grad()
def masked_sequential_sample_neighbors(logits: torch.Tensor, K: int, indeg_cap: int):
    device = logits.device
    N = logits.shape[0]
    probs = torch.softmax(logits, dim=-1)

    neighbors = torch.empty((N, K), dtype=torch.long, device=device)
    indeg = torch.zeros((N,), dtype=torch.long, device=device)
    logp_total = torch.zeros((), device=device)

    for i in range(N):
        p = probs[i].clone()
        p[i] = 0.0
        for k in range(K):
            if indeg_cap is not None and indeg_cap > 0:
                cap_mask = (indeg >= indeg_cap).float()
                p = p * (1.0 - cap_mask)

            s = p.sum()
            if s <= 1e-12:
                avail = torch.ones((N,), device=device)
                avail[i] = 0.0
                if indeg_cap is not None and indeg_cap > 0:
                    avail = avail * (indeg < indeg_cap).float()
                idx = torch.where(avail > 0.0)[0]
                if idx.numel() == 0:
                    idx = torch.tensor([j for j in range(N) if j != i], device=device)
                j = int(idx[torch.randint(0, idx.numel(), (1,), device=device)].item())
                neighbors[i, k] = j
                logp_total = logp_total + torch.log(torch.tensor(1e-12, device=device))
                indeg[j] += 1
                p[j] = 0.0
                continue

            p = p / s
            j = int(torch.distributions.Categorical(p).sample().item())
            neighbors[i, k] = j
            logp_total = logp_total + torch.log(p[j] + 1e-12)
            indeg[j] += 1
            p[j] = 0.0

    return neighbors, logp_total


def evaluate_logprob_neighbors_masked(logits: torch.Tensor, neighbors: torch.Tensor, K: int, indeg_cap: int):
    device = logits.device
    N = logits.shape[0]
    probs = torch.softmax(logits, dim=-1)

    indeg = torch.zeros((N,), dtype=torch.long, device=device)
    logp_total = torch.zeros((), device=device)

    for i in range(N):
        p = probs[i].clone()
        p[i] = 0.0
        for k in range(K):
            if indeg_cap is not None and indeg_cap > 0:
                cap_mask = (indeg >= indeg_cap).float()
                p = p * (1.0 - cap_mask)

            s = p.sum()
            j = int(neighbors[i, k].item())

            if s <= 1e-12:
                logp_total = logp_total + torch.log(torch.tensor(1e-12, device=device))
                indeg[j] += 1
                p[j] = 0.0
                continue

            p = p / s
            logp_total = logp_total + torch.log(p[j] + 1e-12)
            indeg[j] += 1
            p[j] = 0.0

    return logp_total


# =============================================================================
# Gate distribution (kept, default fix_gate=True)
# =============================================================================

def gaussian_logprob(x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * logstd)
    return -0.5 * ((x - mu) ** 2 / (var + 1e-8) + 2.0 * logstd + LOG_2PI)


def sample_squashed_gaussian_gate(mu: torch.Tensor, logstd: torch.Tensor):
    std = torch.exp(logstd)
    z = mu + std * torch.randn_like(mu)
    u = torch.tanh(z)
    gate = 0.5 * (u + 1.0)

    logp_z = gaussian_logprob(z, mu, logstd)
    log_det = torch.log(1.0 - u * u + 1e-6)
    logp_u = logp_z - log_det
    logp_g = logp_u + math.log(2.0)
    logp_gate_norm = logp_g.mean()

    ent = 0.5 * (LOG_2PI + 1.0) + logstd
    ent_gate_mean = ent.mean()
    return gate, logp_gate_norm, ent_gate_mean


def logprob_squashed_gaussian_gate(gate: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
    g = torch.clamp(gate, 1e-6, 1.0 - 1e-6)
    u = 2.0 * g - 1.0
    u = torch.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6)

    z = 0.5 * (torch.log1p(u) - torch.log1p(-u))
    logp_z = gaussian_logprob(z, mu, logstd)
    log_det = torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6)
    logp_u = logp_z - log_det
    logp_g = logp_u + math.log(2.0)
    return logp_g.mean()


# =============================================================================
# Models
# =============================================================================

class GateActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.to_mu = nn.Linear(hidden, 1)
        self.to_logstd_raw = nn.Linear(hidden, 1)

    def forward(self, obs_t: torch.Tensor, logstd_min: float, logstd_max: float):
        h = self.mlp(obs_t)
        mu = self.to_mu(h).squeeze(-1)
        raw = self.to_logstd_raw(h).squeeze(-1)
        s = torch.sigmoid(raw)
        logstd = logstd_min + (logstd_max - logstd_min) * s
        return mu, logstd


class NetActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, emb: int = 64, K: int = 5):
        super().__init__()
        self.K = K
        self.emb = emb
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.to_emb = nn.Linear(hidden, emb)
        self.to_wlogits = nn.Linear(hidden, K)

    def forward(self, obs_t: torch.Tensor, temperature: float = 1.0):
        h = self.mlp(obs_t)
        e = self.to_emb(h)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        base_scale = 1.0 / math.sqrt(e.shape[-1])

        temp = float(temperature)
        logits = temp * (e @ e.t()) * base_scale
        w_logits = self.to_wlogits(h)
        return logits, w_logits


class StepCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        pooled = obs_t.mean(dim=0, keepdim=True)
        return self.v(pooled).squeeze(-1)


class MacroCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        pooled = obs_t.mean(dim=0, keepdim=True)
        return self.v(pooled).squeeze(-1)


# =============================================================================
# Loss helpers
# =============================================================================

def huber_loss(x: torch.Tensor, delta: float):
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device))
    lin = absx - quad
    return 0.5 * quad * quad + delta * lin


def compute_gae(rew, done, v, v_next, discount=0.99, lam=0.95):
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = rew[t] + discount * v_next * mask - v[t]
        gae = delta + discount * lam * mask * gae
        adv[t] = gae
        v_next = v[t]
    ret = adv + v
    return adv, ret


# =============================================================================
# Platform reward (original)
# =============================================================================

def build_info_rewards(b_seq: np.ndarray, R_seq: np.ndarray, info_horizon: int):
    T, _N = b_seq.shape
    H = int(info_horizon)
    r_info = np.zeros((T,), dtype=float)
    if H < 1:
        return r_info

    for t in range(T):
        t1 = min(t + 1, T - 1)
        t2 = min(t + H, T - 1)
        if t2 <= t1:
            Rbar = float(R_seq[t2])
        else:
            Rbar = float(np.mean(R_seq[t1:t2 + 1]))
        err = b_seq[t] - Rbar
        r_info[t] = - float(np.mean(err * err))
    return r_info


def build_platform_rewards(
    b_seq: np.ndarray,
    R_seq: np.ndarray,
    dP_seq: np.ndarray,
    info_horizon: int,
    alpha: float,
    beta: float,
    gamma: float,
    lambda_dP: float,
):
    r_info = build_info_rewards(b_seq=b_seq, R_seq=R_seq, info_horizon=info_horizon)
    r_pol = -np.var(b_seq, axis=1)
    r_rew = -float(lambda_dP) * dP_seq
    r_net = float(alpha) * r_info + float(beta) * r_pol + float(gamma) * r_rew
    return r_net, r_info, r_pol, r_rew


# =============================================================================
# Probe reward (optional)
# =============================================================================

def build_probe_reward_from_batch(
    batch: Dict[str, torch.Tensor],
    lam_risk: float,
    lam_gini: float,
    lam_dP: float,
    risk_clip: float = 50.0,
) -> torch.Tensor:
    risk = torch.clamp(batch["riskS"], 0.0, float(risk_clip))
    gini = torch.clamp(batch["gini"], 0.0, 1.0)
    dP = torch.clamp(batch["dP"], 0.0, 1.0)
    return -(float(lam_risk) * risk + float(lam_gini) * gini + float(lam_dP) * dP)


# =============================================================================
# Rollout
# =============================================================================

def rollout_one(env: InfoNetworkBondEnvFinal,
               gate_actor: GateActor,
               net_actor: NetActor,
               step_critic: StepCritic,
               cfg):

    device = torch.device(cfg.device)
    obs_reg, obs_net = env.reset()

    obs_reg_list, obs_net_list = [], []
    done_list, r_reg_list = [], []
    R_list, b_list, dP_list = [], [], []
    riskS_list, gini_list = [], []
    v_step_list = []

    gate_list, logp_gate_list, ent_gate_list = [], [], []
    netmask_list, neighbors_list, w_list, logp_net_list = [], [], [], []

    dbg_last: Dict[str, Any] = {}
    net_dbg_last = None

    for _ in range(cfg.rollout_len):
        obs_reg_t = torch.tensor(np.asarray(obs_reg, dtype=np.float32), device=device)
        obs_net_t = torch.tensor(np.asarray(obs_net, dtype=np.float32), device=device)

        v_step = step_critic(obs_reg_t)

        if cfg.fix_gate:
            gate = torch.full((env.N,), float(cfg.fixed_gate_value), device=device)
            logp_gate_norm = torch.zeros((), device=device)
            ent_gate_mean = torch.zeros((), device=device)
        else:
            mu, logstd = gate_actor(obs_reg_t, cfg.gate_logstd_min, cfg.gate_logstd_max)
            gate, logp_gate_norm, ent_gate_mean = sample_squashed_gaussian_gate(mu, logstd)

        do_net = ((env.t % cfg.net_period) == 0) and (env.t >= cfg.net_start_t)
        if do_net:
            logits, w_logits = net_actor(obs_net_t, temperature=cfg.net_temperature)
            neighbors, logp_net = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
            w = torch.softmax(w_logits, dim=-1)

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                probs = probs.clone()
                probs.fill_diagonal_(0.0)
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                ent_row = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                top1 = torch.max(probs, dim=-1).values
                net_dbg_last = dict(
                    logits_mean=float(logits.mean().item()),
                    logits_std=float(logits.std(unbiased=False).item()),
                    ent_mean=float(ent_row.mean().item()),
                    ent_std=float(ent_row.std(unbiased=False).item()),
                    top1_mean=float(top1.mean().item()),
                    top1_std=float(top1.std(unbiased=False).item()),
                )
        else:
            neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
            logp_net = torch.zeros((), device=device)
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)

        if do_net:
            (obs_reg, obs_net), r_reg, done, info = env.step(
                g_gate=gate.detach().cpu().numpy(),
                neighbors=neighbors.detach().cpu().numpy(),
                w=w.detach().cpu().numpy(),
            )
        else:
            (obs_reg, obs_net), r_reg, done, info = env.step(
                g_gate=gate.detach().cpu().numpy(),
                neighbors=None,
                w=None,
            )

        obs_reg_list.append(obs_reg_t)
        obs_net_list.append(obs_net_t)
        v_step_list.append(v_step.squeeze())

        done_list.append(float(done))
        r_reg_list.append(float(r_reg))

        R_list.append(float(info["R"]))
        b_list.append(info["b"].copy())
        dP_list.append(float(info["dP"]))
        riskS_list.append(float(info["riskS"]))
        gini_list.append(float(info["gini"]))

        gate_list.append(gate)
        logp_gate_list.append(logp_gate_norm)
        ent_gate_list.append(ent_gate_mean)

        netmask_list.append(float(do_net))
        neighbors_list.append(neighbors)
        w_list.append(w)
        logp_net_list.append(logp_net)

        dbg_last = info
        if done:
            break

    obs_reg_stack = torch.stack(obs_reg_list)
    obs_net_stack = torch.stack(obs_net_list)

    b_seq = np.stack(b_list, axis=0)
    R_seq = np.asarray(R_list, dtype=float)
    dP_seq = np.asarray(dP_list, dtype=float)

    r_net, r_info, r_pol, r_rew = build_platform_rewards(
        b_seq=b_seq,
        R_seq=R_seq,
        dP_seq=dP_seq,
        info_horizon=cfg.info_horizon,
        alpha=cfg.alpha_info,
        beta=cfg.beta_pol,
        gamma=cfg.gamma_rew,
        lambda_dP=cfg.lambda_dP,
    )

    batch = {
        "obs_reg": obs_reg_stack,
        "obs_net": obs_net_stack,
        "done": torch.tensor(done_list, dtype=torch.float32, device=device),

        "r_reg": torch.tensor(r_reg_list, dtype=torch.float32, device=device),
        "r_net_orig": torch.tensor(r_net, dtype=torch.float32, device=device),
        "r_info": torch.tensor(r_info, dtype=torch.float32, device=device),
        "r_pol": torch.tensor(r_pol, dtype=torch.float32, device=device),
        "r_rew": torch.tensor(r_rew, dtype=torch.float32, device=device),

        "dP": torch.tensor(dP_list, dtype=torch.float32, device=device),
        "riskS": torch.tensor(riskS_list, dtype=torch.float32, device=device),
        "gini": torch.tensor(gini_list, dtype=torch.float32, device=device),

        "v_step": torch.stack(v_step_list),

        "gate": torch.stack(gate_list),
        "logp_gate": torch.stack(logp_gate_list),
        "ent_gate": torch.stack(ent_gate_list),

        "netmask": torch.tensor(netmask_list, dtype=torch.float32, device=device),
        "neighbors": neighbors_list,
        "w": w_list,
        "logp_net": torch.stack(logp_net_list),

        "dbg_last": dbg_last,
        "net_dbg_last": net_dbg_last,
    }
    return batch


# =============================================================================
# Macro builder for Net-PPO
# =============================================================================

def build_macro_sequence_from_step_reward(step_rew: torch.Tensor, done: torch.Tensor, netmask: torch.Tensor,
                                         net_period: int, discount: float):
    T = step_rew.shape[0]
    idx = torch.where(netmask > 0.5)[0]
    if idx.numel() == 0:
        return {
            "idx_net": idx,
            "r_macro": torch.zeros((0,), device=step_rew.device),
            "done_macro": torch.zeros((0,), device=step_rew.device),
        }

    r_macro = torch.zeros((idx.numel(),), device=step_rew.device)
    done_macro = torch.zeros((idx.numel(),), device=step_rew.device)

    for m, t in enumerate(idx.tolist()):
        s = 0.0
        last = t
        for k in range(net_period):
            tt = t + k
            if tt >= T:
                break
            s += (discount ** k) * float(step_rew[tt].item())
            last = tt
            if done[tt].item() > 0.5:
                break
        r_macro[m] = float(s)
        done_macro[m] = float(done[last].item())

    return {"idx_net": idx, "r_macro": r_macro, "done_macro": done_macro}


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    topology: str = "random_fixed"
    beta_small_world: float = 0.1
    topology_seed: int = 0
    env_seed: int = 0
    train_seed: int = 0

    horizon: int = 1000
    N: int = 50
    K: int = 5
    indeg_cap: int = 8
    net_period: int = 5
    net_start_t: int = 5

    gamma_fin: float = 5.0

    rollout_len: int = 200
    n_iters: int = 200

    info_horizon: int = 10
    alpha_info: float = 10.0
    beta_pol: float = 1.0
    gamma_rew: float = 0.1
    lambda_dP: float = 0.1

    net_temperature: float = 1.0
    net_ent_coef: float = 0.0

    lr_net: float = 5e-4
    net_clip: float = 0.2
    net_mini_epochs: int = 5
    net_minibatch: int = 32
    net_discount: float = 0.99
    net_gae_lambda: float = 0.95
    net_logp_scale: float = 1.0

    lr_macro_critic: float = 5e-4
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    ret_clip_macro: float = 50.0
    huber_delta: float = 10.0

    train_net: bool = True

    fix_gate: bool = True
    fixed_gate_value: float = 0.4
    train_gate: bool = False
    gate_logstd_min: float = -2.5
    gate_logstd_max: float = -0.5

    use_probe_reward: bool = False
    probe_lam_risk: float = 0.05
    probe_lam_gini: float = 2.0
    probe_lam_dP: float = 1.0
    probe_risk_clip: float = 50.0

    device: str = "cpu"


# =============================================================================
# Train
# =============================================================================

def train(cfg: Config):
    device = torch.device(cfg.device)
    set_seed(cfg.train_seed)

    env = InfoNetworkBondEnvFinal(
        topology=cfg.topology,
        beta_small_world=cfg.beta_small_world,
        topology_seed=cfg.topology_seed,
        seed=cfg.env_seed,
        N=cfg.N,
        K=cfg.K,
        net_period=cfg.net_period,
        indeg_cap=cfg.indeg_cap,
        horizon=cfg.horizon,
    )
    env.gamma_fin = cfg.gamma_fin

    obs_reg0, obs_net0 = env.reset()
    obs_dim_reg = len(obs_reg0[0])
    obs_dim_net = len(obs_net0[0])

    gate_actor = GateActor(obs_dim=obs_dim_reg, hidden=128).to(device)
    net_actor = NetActor(obs_dim=obs_dim_net, hidden=128, emb=64, K=cfg.K).to(device)
    step_critic = StepCritic(obs_dim=obs_dim_reg, hidden=128).to(device)
    macro_critic = MacroCritic(obs_dim=obs_dim_net, hidden=128).to(device)

    opt_net = optim.Adam(net_actor.parameters(), lr=cfg.lr_net)
    opt_v_macro = optim.Adam(macro_critic.parameters(), lr=cfg.lr_macro_critic)

    print("=" * 118)
    print("NetPPO-CausalProbe v10.11")
    print(f"TOPOLOGY={cfg.topology} beta={cfg.beta_small_world} | topo_seed={cfg.topology_seed} env_seed={cfg.env_seed} train_seed={cfg.train_seed}")
    print(f"N={cfg.N} K={cfg.K} indeg_cap={cfg.indeg_cap} net_period={cfg.net_period} net_start_t={cfg.net_start_t} horizon={cfg.horizon} rollout_len={cfg.rollout_len}")
    print(f"Net: temp={cfg.net_temperature} ent_coef={cfg.net_ent_coef} | lr_net={cfg.lr_net} clip={cfg.net_clip} epochs={cfg.net_mini_epochs} mb={cfg.net_minibatch}")
    print(f"Reward: {'PROBE' if cfg.use_probe_reward else 'ORIGINAL'} | train_net={cfg.train_net} fix_gate={cfg.fix_gate}({cfg.fixed_gate_value})")
    if cfg.use_probe_reward:
        print(f"Probe: lam_risk={cfg.probe_lam_risk} lam_gini={cfg.probe_lam_gini} lam_dP={cfg.probe_lam_dP} risk_clip={cfg.probe_risk_clip}")
    else:
        print("Original platform reward: alpha*info + beta*(-Var(b)) + gamma*(-lambda_dP*dP)")
        print(f"alpha={cfg.alpha_info} beta={cfg.beta_pol} gamma={cfg.gamma_rew} lambda_dP={cfg.lambda_dP} H={cfg.info_horizon}")
    print("=" * 118)

    for it in range(cfg.n_iters):
        batch = rollout_one(env, gate_actor, net_actor, step_critic, cfg)

        if cfg.use_probe_reward:
            r_net = build_probe_reward_from_batch(
                batch,
                lam_risk=cfg.probe_lam_risk,
                lam_gini=cfg.probe_lam_gini,
                lam_dP=cfg.probe_lam_dP,
                risk_clip=cfg.probe_risk_clip,
            )
        else:
            r_net = batch["r_net_orig"]

        obs_net = batch["obs_net"]
        done = batch["done"]
        netmask = batch["netmask"]

        macro = build_macro_sequence_from_step_reward(
            step_rew=r_net,
            done=done,
            netmask=netmask,
            net_period=cfg.net_period,
            discount=cfg.net_discount,
        )
        idx_net = macro["idx_net"]
        M = idx_net.numel()

        net_ratio_means, net_kls, net_clipfracs = [], [], []
        dlogp_list = []

        if cfg.train_net and M > 0:
            obs_net_at = obs_net[idx_net]

            with torch.no_grad():
                v_macro = torch.zeros((M,), device=device)
                for m in range(M):
                    v_macro[m] = macro_critic(obs_net_at[m])
                v_macro_next = v_macro[-1] * (1.0 - macro["done_macro"][-1])

            adv_net, ret_net = compute_gae(
                rew=macro["r_macro"],
                done=macro["done_macro"],
                v=v_macro.detach(),
                v_next=v_macro_next.detach(),
                discount=cfg.net_discount,
                lam=cfg.net_gae_lambda,
            )
            adv_net = (adv_net - adv_net.mean()) / (adv_net.std(unbiased=False) + 1e-8)
            ret_net = torch.clamp(ret_net, -cfg.ret_clip_macro, cfg.ret_clip_macro)

            logp_net_old = batch["logp_net"][idx_net].detach() * cfg.net_logp_scale
            neighbors_old = [batch["neighbors"][t] for t in idx_net.tolist()]

            midx = torch.arange(M, device=device)

            for _ in range(cfg.net_mini_epochs):
                perm = midx[torch.randperm(M)]
                for start in range(0, M, cfg.net_minibatch):
                    mbm = perm[start:start + cfg.net_minibatch]
                    if mbm.numel() == 0:
                        continue

                    logp_new_list = []
                    v_mb = torch.zeros((mbm.numel(),), device=device)
                    ent_mb_list = []

                    for j, mm in enumerate(mbm.tolist()):
                        obs_t = obs_net_at[mm]
                        v_mb[j] = macro_critic(obs_t)

                        logits_t, _ = net_actor(obs_t, temperature=cfg.net_temperature)

                        lp = evaluate_logprob_neighbors_masked(
                            logits=logits_t,
                            neighbors=neighbors_old[mm],
                            K=cfg.K,
                            indeg_cap=cfg.indeg_cap,
                        ) * cfg.net_logp_scale
                        logp_new_list.append(lp)

                        if cfg.net_ent_coef > 0:
                            with torch.no_grad():
                                probs = torch.softmax(logits_t, dim=-1)
                                probs = probs.clone()
                                probs.fill_diagonal_(0.0)
                                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                                ent_row = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                                ent_mb_list.append(ent_row.mean())

                    logp_net_new = torch.stack(logp_new_list)

                    if cfg.net_ent_coef > 0 and ent_mb_list:
                        ent_mb = torch.stack(ent_mb_list).mean()
                    else:
                        ent_mb = torch.zeros((), device=device)

                    with torch.no_grad():
                        dlogp_list.append(float(torch.mean(torch.abs(logp_net_new - logp_net_old[mbm])).item()))

                    ratio = torch.exp(logp_net_new - logp_net_old[mbm])
                    surr1 = ratio * adv_net[mbm]
                    surr2 = torch.clamp(ratio, 1.0 - cfg.net_clip, 1.0 + cfg.net_clip) * adv_net[mbm]
                    ppo_loss = -torch.mean(torch.min(surr1, surr2))

                    net_loss = ppo_loss - float(cfg.net_ent_coef) * ent_mb

                    td = v_mb - ret_net[mbm]
                    v_loss = torch.mean(huber_loss(td, cfg.huber_delta))

                    opt_net.zero_grad(set_to_none=True)
                    net_loss.backward()
                    nn.utils.clip_grad_norm_(net_actor.parameters(), cfg.max_grad_norm)
                    opt_net.step()

                    opt_v_macro.zero_grad(set_to_none=True)
                    (cfg.vf_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(macro_critic.parameters(), cfg.max_grad_norm)
                    opt_v_macro.step()

                    with torch.no_grad():
                        approx_kl = torch.mean(logp_net_old[mbm] - logp_net_new)
                        clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.net_clip).float())

                    net_ratio_means.append(float(ratio.mean().item()))
                    net_kls.append(float(approx_kl.item()))
                    net_clipfracs.append(float(clipfrac.item()))

        if it % 10 == 0:
            dP_all = batch["dP"].detach().cpu().numpy()
            nm = batch["netmask"].detach().cpu().numpy() > 0.5
            dP_mean_all = float(dP_all.mean()) if dP_all.size else 0.0
            dP_mean_net = float(dP_all[nm].mean()) if nm.any() else 0.0
            dP_nz_net = int((dP_all[nm] > 1e-12).sum()) if nm.any() else 0
            dP_cnt_net = int(nm.sum())

            nd = batch.get("net_dbg_last", None) or {}
            dbg_last = batch["dbg_last"]

            rnet_mean = float(r_net.mean().item())
            rreg_mean = float(batch["r_reg"].mean().item())
            rreg_std = float(batch["r_reg"].std(unbiased=False).item())

            print(
                f"iter={it:04d} | "
                f"r_reg={rreg_mean:+.4f}±{rreg_std:.4f} | "
                f"r_net_used={rnet_mean:+.4f} | "
                f"NET(M={M}): ratio={np.mean(net_ratio_means) if net_ratio_means else 1.0:.3f} "
                f"kl={np.mean(net_kls) if net_kls else 0.0:+.4f} "
                f"clip={np.mean(net_clipfracs) if net_clipfracs else 0.0:.3f} "
                f"dlogp={np.mean(dlogp_list) if dlogp_list else 0.0:.4f} | "
                f"logits_mu={nd.get('logits_mean',0.0):+.3f} logits_std={nd.get('logits_std',0.0):.3f} "
                f"ent={nd.get('ent_mean',0.0):.3f}±{nd.get('ent_std',0.0):.3f} "
                f"top1={nd.get('top1_mean',0.0):.4f}±{nd.get('top1_std',0.0):.4f} | "
                f"dP_mean_all={dP_mean_all:.6f} dP_mean_net={dP_mean_net:.6f} dP_nz_net={dP_nz_net}/{dP_cnt_net} | "
                f"dbg_last[dP]={dbg_last.get('dP',0.0):.6f} dbg_last[gini]={dbg_last.get('gini',0.0):.3f} dbg_last[riskS]={dbg_last.get('riskS',0.0):.3g}"
            )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", type=str, default="random_fixed",
                   choices=["random_fixed", "scale_free", "small_world"])
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--topo_seed", type=int, default=None)
    p.add_argument("--env_seed", type=int, default=None)
    p.add_argument("--train_seed", type=int, default=None)

    p.add_argument("--net_temperature", type=float, default=1.0)
    p.add_argument("--net_ent_coef", type=float, default=0.0)
    p.add_argument("--net_start_t", type=int, default=None)

    p.add_argument("--n_iters", type=int, default=200)
    p.add_argument("--rollout_len", type=int, default=200)

    p.add_argument("--use_probe_reward", type=int, default=0)
    p.add_argument("--probe_lam_risk", type=float, default=0.05)
    p.add_argument("--probe_lam_gini", type=float, default=2.0)
    p.add_argument("--probe_lam_dP", type=float, default=1.0)
    p.add_argument("--probe_risk_clip", type=float, default=50.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = int(args.seed)

    net_period_default = 5
    net_start_t = int(args.net_start_t) if args.net_start_t is not None else net_period_default

    cfg = Config(
        topology=args.topology,
        beta_small_world=float(args.beta),

        topology_seed=base if args.topo_seed is None else int(args.topo_seed),
        env_seed=base if args.env_seed is None else int(args.env_seed),
        train_seed=base if args.train_seed is None else int(args.train_seed),

        net_temperature=float(args.net_temperature),
        net_ent_coef=float(args.net_ent_coef),
        net_start_t=int(net_start_t),

        n_iters=int(args.n_iters),
        rollout_len=int(args.rollout_len),

        use_probe_reward=bool(int(args.use_probe_reward)),
        probe_lam_risk=float(args.probe_lam_risk),
        probe_lam_gini=float(args.probe_lam_gini),
        probe_lam_dP=float(args.probe_lam_dP),
        probe_risk_clip=float(args.probe_risk_clip),
    )

    if cfg.net_start_t <= 0:
        cfg.net_start_t = cfg.net_period

    train(cfg)