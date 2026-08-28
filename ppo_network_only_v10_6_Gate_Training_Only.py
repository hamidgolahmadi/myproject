# -*- coding: utf-8 -*-
"""
ppo_network_only_v10_7_modular.py

v10.7 (MODULAR TRAINING MODES) — Project A clean separation + proper freeze controls

Core story:
- GateActor = Regulator/Market-stabilizer (LIMITED obs, reward r_reg for stability)
- NetActor  = Platform/Market (RICH obs, reward r_net for info-quality & anti-polarization & rewiring cost)

NEW (this file):
- Explicit training switches:
    TRAIN_GATE / TRAIN_NET
- Explicit action freezes:
    FIX_GATE (constant g_i) OR learn gate
    FIX_NET  (no rewiring action applied; network stays fixed)
- When TRAIN_* is False, the ENTIRE corresponding PPO+critic block is skipped (no updates).
- Debug:
    ||Δθ_net||_2 and max|Δθ_net| (always computed; should be ~0 when TRAIN_NET=False)
    logits mean/std and neighbor-entropy mean/std on a net step (if any)

IMPORTANT:
- If you set FIX_GATE=True, you should also set TRAIN_GATE=False
  (otherwise you’re “training” a policy that never acts).
- If you set FIX_NET=True, you should also set TRAIN_NET=False
  (otherwise you’re training net policy but it never changes the environment).

"""

import math
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

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


# =============================================================================
# Environment
# =============================================================================

class InfoNetworkBondEnvV10_7:
    """
    Mechanics:
      - Beliefs: private signal + DeGroot diffusion via P
      - Price formation: impact from aggregate flow + noise
      - Rewiring: applied every net_period steps (when provided)

    Observations:
      - obs_reg: LIMITED (public-ish)
      - obs_net: RICH (platform/agents can see more)

    Rewards inside env:
      - r_reg (stability shaping): risk/flow/gini/dP (NO info-quality here)
      - r_net computed OUTSIDE env (needs multi-step return target).
    """

    def __init__(
        self,
        N=50,
        K=5,
        net_period=5,
        indeg_cap=8,
        horizon=1000,
        p0=100.0,
        kappa=0.02,
        sigma_eps=0.1,
        x_max=1.0,
        tau=0.001,
        rho_y=0.98,
        sigma_y=0.02,
        sigma_s=0.05,
        omega_social=0.7,
        sigma_belief=0.02,
        beta_risk=None,
        # stability shaping weights (regulator reward)
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=0.5,
        w_net=0.05,
        cap_riskS=50.0,
        cap_flow4=5.0,
        cap_reward=50.0,
        seed=0,
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

        self.cap_riskS = cap_riskS
        self.cap_flow4 = cap_flow4
        self.cap_reward = cap_reward

        self.rng = np.random.default_rng(seed)

        # runtime-set by trainer
        self.gamma_fin = 5.0

        self.reset()

    def _init_P_random_topk(self):
        P = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            candidates = [j for j in range(self.N) if j != i]
            nbrs = self.rng.choice(candidates, size=self.K, replace=False)
            w = self.rng.random(self.K)
            w = w / w.sum()
            P[i, nbrs] = w
        return P

    def _private_signals(self):
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

        self.P = self._init_P_random_topk()
        self.P_prev = self.P.copy()

        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.s = self._private_signals()

        self.b = self.rng.normal(0.0, 0.2, size=self.N)

        self.risk_v = 0.0
        self.delta_x_prev = np.zeros(self.N, dtype=float)

        return self._build_obs_reg(), self._build_obs_net()

    def step(self, g_gate, neighbors=None, w=None):
        # Rewire on net steps only if provided
        if (self.t % self.net_period) == 0 and neighbors is not None and w is not None:
            self.P_prev = self.P.copy()
            self.P = row_stochastic_from_neighbors(self.N, neighbors, w)

        # latent
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        # belief update
        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        # finance action (gate dampens reaction to social signal)
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

        # risk
        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        # cash
        self.c = self.c - self.p * delta_x - tc
        self.delta_x_prev = delta_x.copy()

        # ---------- regulator reward (stability shaping) ----------
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


# =============================================================================
# Network sampling (indegree cap)
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
# Squashed Gaussian gate
# =============================================================================

LOG_2PI = float(math.log(2.0 * math.pi))

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
    logp_gate_norm = logp_g.mean()  # mean over agents

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
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.to_emb = nn.Linear(hidden, emb)
        self.to_wlogits = nn.Linear(hidden, K)

    def forward(self, obs_t: torch.Tensor):
        h = self.mlp(obs_t)
        e = self.to_emb(h)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        scale = 1.0 / math.sqrt(e.shape[-1])
        logits = (e @ e.t()) * scale  # (N,N)
        w_logits = self.to_wlogits(h) # (N,K)
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
# GAE + losses
# =============================================================================

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

def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().data.norm(2).item()
        total += g * g
    return float(math.sqrt(total))

def huber_loss(x: torch.Tensor, delta: float):
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device))
    lin = absx - quad
    return 0.5 * quad * quad + delta * lin


# =============================================================================
# Platform reward components (computed outside env)
# =============================================================================

def build_info_rewards(b_seq: np.ndarray, R_seq: np.ndarray, info_horizon: int):
    """
    Rbar_t = mean_{h=1..H} R_{t+h}
    r_info_t = - mean_i ( b_i(t) - Rbar_t )^2
    """
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
    """
    r_net = alpha * r_info + beta * r_pol + gamma * r_rew

    r_info = - MSE_i (b_i(t) - Rbar_{t:t+H})
    r_pol  = - Var(b(t))              (anti-polarization)
    r_rew  = - lambda_dP * dP_t       (rewiring cost / network jitter cost)
    """
    r_info = build_info_rewards(b_seq=b_seq, R_seq=R_seq, info_horizon=info_horizon)
    r_pol  = - np.var(b_seq, axis=1)  # per-timestep var across agents
    r_rew  = - float(lambda_dP) * dP_seq
    r_net = float(alpha) * r_info + float(beta) * r_pol + float(gamma) * r_rew
    return r_net, r_info, r_pol, r_rew


# =============================================================================
# Rollout (supports FIX_GATE and FIX_NET)
# =============================================================================

def rollout_one(
    env: InfoNetworkBondEnvV10_7,
    gate_actor: GateActor,
    net_actor: NetActor,
    step_critic: StepCritic,
    cfg,
    *,
    FIX_GATE: bool,
    FIX_GATE_VALUE: float,
    FIX_NET: bool,
):
    device = cfg.device

    obs_reg, obs_net = env.reset()

    obs_reg_list, obs_net_list = [], []
    done_list = []
    r_reg_list = []

    # For platform reward construction
    R_list, b_list, dP_list = [], [], []

    v_step_list = []

    gate_list, logp_gate_list, ent_gate_list = [], [], []
    netmask_list, neighbors_list, w_list, logp_net_list = [], [], [], []

    dbg_last = {}
    net_dbg_last = None  # logits/entropy stats for ONE net step (last seen)

    for _ in range(cfg.rollout_len):
        obs_reg_t = torch.tensor(np.asarray(obs_reg, dtype=np.float32), device=device)
        obs_net_t = torch.tensor(np.asarray(obs_net, dtype=np.float32), device=device)

        v_step = step_critic(obs_reg_t)

        # -----------------
        # Gate action
        # -----------------
        if FIX_GATE:
            gate = torch.full((env.N,), float(FIX_GATE_VALUE), device=device)
            logp_gate_norm = torch.zeros((), device=device)  # dummy
            ent_gate_mean  = torch.zeros((), device=device)  # dummy
        else:
            mu, logstd = gate_actor(obs_reg_t, cfg.gate_logstd_min, cfg.gate_logstd_max)
            gate, logp_gate_norm, ent_gate_mean = sample_squashed_gaussian_gate(mu, logstd)

        # -----------------
        # Net action (rewiring)
        # -----------------
        do_net = (env.t % cfg.net_period) == 0

        if do_net and (not FIX_NET):
            logits, w_logits = net_actor(obs_net_t)
            neighbors, logp_net = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
            w = torch.softmax(w_logits, dim=-1)

            # debug stats for one net step
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                probs = probs.clone()
                probs.fill_diagonal_(0.0)
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                ent_row = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                net_dbg_last = dict(
                    logits_mean=float(logits.mean().item()),
                    logits_std=float(logits.std(unbiased=False).item()),
                    ent_mean=float(ent_row.mean().item()),
                    ent_std=float(ent_row.std(unbiased=False).item()),
                )
        else:
            # Either not a net step, OR net is fixed (no rewiring action).
            neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
            logp_net = torch.zeros((), device=device)
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)

        # -----------------
        # Step env
        # -----------------
        if do_net and (not FIX_NET):
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

        gate_list.append(gate)
        logp_gate_list.append(logp_gate_norm)
        ent_gate_list.append(ent_gate_mean)

        netmask_list.append(float(do_net and (not FIX_NET)))  # IMPORTANT: netmask=1 only when rewiring is actually active
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
        "r_net": torch.tensor(r_net, dtype=torch.float32, device=device),
        "r_info": torch.tensor(r_info, dtype=torch.float32, device=device),
        "r_pol": torch.tensor(r_pol, dtype=torch.float32, device=device),
        "r_rew": torch.tensor(r_rew, dtype=torch.float32, device=device),

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
# Macro builder for Net-PPO (aggregates r_net)
# =============================================================================

def build_macro_sequence_from_step_reward(
    step_rew: torch.Tensor,
    done: torch.Tensor,
    netmask: torch.Tensor,
    net_period: int,
    discount: float,
):
    T = step_rew.shape[0]
    idx = torch.where(netmask > 0.5)[0]
    if idx.numel() == 0:
        return {"idx_net": idx,
                "r_macro": torch.zeros((0,), device=step_rew.device),
                "done_macro": torch.zeros((0,), device=step_rew.device)}

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
class ConfigV10_7:
    horizon: int = 1000
    N: int = 50
    K: int = 5
    indeg_cap: int = 8
    net_period: int = 5
    gamma_fin: float = 5.0

    rollout_len: int = 200
    n_iters: int = 200

    # Platform objective weights
    info_horizon: int = 10
    alpha_info: float = 10.0
    beta_pol: float = 1.0
    gamma_rew: float = 1.0
    lambda_dP: float = 1.0

    # Gate PPO (regulator)
    lr_gate: float = 5e-5
    gate_clip: float = 0.1
    gate_mini_epochs: int = 2
    gate_minibatch: int = 64
    gate_discount: float = 0.99
    gate_gae_lambda: float = 0.95
    gate_ent_coef: float = 0.005
    gate_logstd_min: float = -2.5
    gate_logstd_max: float = -0.5
    w_gate_mean: float = 0.2
    gate_target: float = 0.4

    # Net PPO (platform)
    lr_net: float = 5e-4
    net_clip: float = 0.2
    net_mini_epochs: int = 5
    net_minibatch: int = 32
    net_discount: float = 0.99
    net_gae_lambda: float = 0.95
    net_logp_scale: float = 1.0

    # Critics
    lr_step_critic: float = 5e-4
    lr_macro_critic: float = 5e-4
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    ret_clip_step: float = 20.0
    ret_clip_macro: float = 50.0
    huber_delta: float = 10.0

    device: str = "cpu"


# =============================================================================
# Training (MODULAR)
# =============================================================================

def train_v10_7():
    cfg = ConfigV10_7()
    device = torch.device(cfg.device)

    # ============================
    # TRAINING / FREEZE SWITCHES
    # ============================
    # --- choose ONE mode ---
    # Mode A: Train NET only, gate fixed
    TRAIN_GATE = False
    TRAIN_NET  = True
    FIX_GATE = True
    FIX_GATE_VALUE = 0.4
    FIX_NET = False

    # Mode B: Train GATE only, network fixed
    # TRAIN_GATE = True
    # TRAIN_NET  = False
    # FIX_GATE = False
    # FIX_GATE_VALUE = 0.4
    # FIX_NET = True

    # Mode C: Train BOTH (no freezes)
    # TRAIN_GATE = True
    # TRAIN_NET  = True
    # FIX_GATE = False
    # FIX_GATE_VALUE = 0.4
    # FIX_NET = False

    if FIX_GATE and TRAIN_GATE:
        print("WARNING: FIX_GATE=True but TRAIN_GATE=True. You are updating a policy that never acts. Set TRAIN_GATE=False.")
    if FIX_NET and TRAIN_NET:
        print("WARNING: FIX_NET=True but TRAIN_NET=True. You are updating NetActor but rewiring is disabled. Set TRAIN_NET=False.")

    env_kwargs = dict(
        N=cfg.N, K=cfg.K, net_period=cfg.net_period, indeg_cap=cfg.indeg_cap, horizon=cfg.horizon,
        p0=100.0, kappa=0.02, sigma_eps=0.1, x_max=1.0, tau=0.001,
        rho_y=0.98, sigma_y=0.02, sigma_s=0.05, omega_social=0.7, sigma_belief=0.02,
        risk_unit=1e-6,
        w_risk=1.0, w_flow2=0.25, w_flow4=1.0, w_gini=0.5,
        w_net=0.05,
        cap_riskS=50.0, cap_flow4=5.0, cap_reward=50.0,
    )

    print("=" * 118)
    print(f"CONFIG v10.7: N={cfg.N} horizon={cfg.horizon} | gamma_fin={cfg.gamma_fin} | K={cfg.K} indeg_cap={cfg.indeg_cap} net_period={cfg.net_period}")
    print("Obs split: obs_reg (LIMITED) for GateActor | obs_net (RICH) for NetActor")
    print(f"Platform reward: alpha*info + beta*(-Var(b)) + gamma*(-lambda_dP*dP)")
    print(f"alpha={cfg.alpha_info} beta={cfg.beta_pol} gamma={cfg.gamma_rew} lambda_dP={cfg.lambda_dP} H={cfg.info_horizon}")
    print(f"MODE: TRAIN_GATE={TRAIN_GATE} TRAIN_NET={TRAIN_NET} | FIX_GATE={FIX_GATE}({FIX_GATE_VALUE}) FIX_NET={FIX_NET}")
    print("=" * 118)

    seed_train = 0
    set_seed(seed_train)

    env = InfoNetworkBondEnvV10_7(seed=seed_train, **env_kwargs)
    env.gamma_fin = cfg.gamma_fin

    obs_reg0, obs_net0 = env.reset()
    obs_dim_reg = len(obs_reg0[0])
    obs_dim_net = len(obs_net0[0])

    gate_actor = GateActor(obs_dim=obs_dim_reg, hidden=128).to(device)
    net_actor  = NetActor(obs_dim=obs_dim_net, hidden=128, emb=64, K=cfg.K).to(device)

    step_critic  = StepCritic(obs_dim=obs_dim_reg, hidden=128).to(device)
    macro_critic = MacroCritic(obs_dim=obs_dim_net, hidden=128).to(device)

    opt_gate = optim.Adam(gate_actor.parameters(), lr=cfg.lr_gate)
    opt_net  = optim.Adam(net_actor.parameters(),  lr=cfg.lr_net)

    opt_v_step  = optim.Adam(step_critic.parameters(),  lr=cfg.lr_step_critic)
    opt_v_macro = optim.Adam(macro_critic.parameters(), lr=cfg.lr_macro_critic)

    for it in range(cfg.n_iters):
        batch = rollout_one(
            env, gate_actor, net_actor, step_critic, cfg,
            FIX_GATE=FIX_GATE, FIX_GATE_VALUE=FIX_GATE_VALUE, FIX_NET=FIX_NET
        )

        obs_reg = batch["obs_reg"]
        obs_net = batch["obs_net"]
        done = batch["done"]
        T = obs_reg.shape[0]

        # ============================
        # (A) Gate-PPO on r_reg  (ONLY if TRAIN_GATE)
        # ============================
        gate_ratio_means, gate_ratio_stds, gate_kls, gate_clipfracs = [], [], [], []
        ent_gate_means = []
        gate_loss_list, vstep_loss_list = [], []
        gn_gate_list, gn_vstep_list = [], []
        gate_mu_stats, gate_ls_stats, gate_g_stats = [], [], []

        if TRAIN_GATE:
            r_reg = batch["r_reg"]
            v_step = batch["v_step"].detach()
            with torch.no_grad():
                v_next = step_critic(obs_reg[-1]) * (1.0 - done[-1])

            adv_gate, ret_gate = compute_gae(
                rew=r_reg, done=done, v=v_step, v_next=v_next,
                discount=cfg.gate_discount, lam=cfg.gate_gae_lambda,
            )
            adv_gate = (adv_gate - adv_gate.mean()) / (adv_gate.std(unbiased=False) + 1e-8)
            ret_gate = torch.clamp(ret_gate, -cfg.ret_clip_step, cfg.ret_clip_step)

            logp_gate_old = batch["logp_gate"].detach()
            gate_old = batch["gate"].detach()

            idx = torch.arange(T, device=device)

            for _ in range(cfg.gate_mini_epochs):
                perm = idx[torch.randperm(T)]
                for start in range(0, T, cfg.gate_minibatch):
                    mb = perm[start:start + cfg.gate_minibatch]
                    if mb.numel() == 0:
                        continue

                    logp_new_list = []
                    ent_list = []
                    v_mb = torch.zeros((mb.numel(),), device=device)

                    mu_cat, ls_cat, g_cat = [], [], []

                    for j, t in enumerate(mb.tolist()):
                        obs_t = obs_reg[t]
                        v_mb[j] = step_critic(obs_t)

                        mu_t, logstd_t = gate_actor(obs_t, cfg.gate_logstd_min, cfg.gate_logstd_max)
                        lp = logprob_squashed_gaussian_gate(gate_old[t], mu_t, logstd_t)
                        logp_new_list.append(lp)

                        ent = (0.5 * (LOG_2PI + 1.0) + logstd_t).mean()
                        ent_list.append(ent)

                        mu_cat.append(mu_t.detach())
                        ls_cat.append(logstd_t.detach())
                        g_cat.append(gate_old[t].detach())

                    logp_gate_new = torch.stack(logp_new_list)
                    ent_gate = torch.stack(ent_list)

                    ratio = torch.exp(logp_gate_new - logp_gate_old[mb])
                    surr1 = ratio * adv_gate[mb]
                    surr2 = torch.clamp(ratio, 1.0 - cfg.gate_clip, 1.0 + cfg.gate_clip) * adv_gate[mb]
                    pg_loss = -torch.mean(torch.min(surr1, surr2))

                    g_mean_mb = gate_old[mb].mean()
                    gate_mean_pen = cfg.w_gate_mean * (g_mean_mb - cfg.gate_target) ** 2

                    ent_bonus = torch.mean(ent_gate)
                    gate_loss = pg_loss - cfg.gate_ent_coef * ent_bonus + gate_mean_pen

                    td = v_mb - ret_gate[mb]
                    v_loss = torch.mean(huber_loss(td, cfg.huber_delta))

                    opt_gate.zero_grad(set_to_none=True)
                    gate_loss.backward()
                    nn.utils.clip_grad_norm_(gate_actor.parameters(), cfg.max_grad_norm)
                    opt_gate.step()

                    opt_v_step.zero_grad(set_to_none=True)
                    (cfg.vf_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(step_critic.parameters(), cfg.max_grad_norm)
                    opt_v_step.step()

                    with torch.no_grad():
                        approx_kl = torch.mean(logp_gate_old[mb] - logp_gate_new)
                        clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.gate_clip).float())

                    gate_loss_list.append(float(gate_loss.item()))
                    vstep_loss_list.append(float(v_loss.item()))
                    gate_ratio_means.append(float(ratio.mean().item()))
                    gate_ratio_stds.append(float(ratio.std(unbiased=False).item()))
                    gate_kls.append(float(approx_kl.item()))
                    gate_clipfracs.append(float(clipfrac.item()))
                    ent_gate_means.append(float(ent_bonus.item()))
                    gn_gate_list.append(float(grad_norm(gate_actor)))
                    gn_vstep_list.append(float(grad_norm(step_critic)))

                    mu_cat = torch.cat(mu_cat, dim=0)
                    ls_cat = torch.cat(ls_cat, dim=0)
                    g_cat = torch.cat(g_cat, dim=0)
                    gate_mu_stats.append((float(mu_cat.mean().item()), float(mu_cat.std(unbiased=False).item())))
                    gate_ls_stats.append((float(ls_cat.mean().item()), float(ls_cat.std(unbiased=False).item())))
                    gate_g_stats.append((float(g_cat.mean().item()), float(g_cat.std(unbiased=False).item())))

        # ============================
        # (B) Net-PPO on r_net (macro) (ONLY if TRAIN_NET)
        # ============================
        # Freeze diagnostics snapshots (BEFORE net update)
        with torch.no_grad():
            net_params_before = [p.detach().clone() for p in net_actor.parameters()]

        net_ratio_means, net_ratio_stds, net_kls, net_clipfracs = [], [], [], []
        net_loss_list, vmacro_loss_list = [], []
        gn_net_list, gn_vmacro_list = [], []
        dlogp_list = []

        M = 0
        if TRAIN_NET:
            r_net = batch["r_net"]
            netmask = batch["netmask"]

            macro = build_macro_sequence_from_step_reward(
                step_rew=r_net, done=done, netmask=netmask,
                net_period=cfg.net_period, discount=cfg.net_discount
            )
            idx_net = macro["idx_net"]
            M = idx_net.numel()

            if M > 0:
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

                        for j, mm in enumerate(mbm.tolist()):
                            obs_t = obs_net_at[mm]
                            v_mb[j] = macro_critic(obs_t)

                            logits_t, _ = net_actor(obs_t)
                            lp = evaluate_logprob_neighbors_masked(
                                logits=logits_t,
                                neighbors=neighbors_old[mm],
                                K=cfg.K,
                                indeg_cap=cfg.indeg_cap,
                            )
                            lp = lp * cfg.net_logp_scale
                            logp_new_list.append(lp)

                        logp_net_new = torch.stack(logp_new_list)

                        with torch.no_grad():
                            dlogp = torch.mean(torch.abs(logp_net_new - logp_net_old[mbm]))
                            dlogp_list.append(float(dlogp.item()))

                        ratio = torch.exp(logp_net_new - logp_net_old[mbm])
                        surr1 = ratio * adv_net[mbm]
                        surr2 = torch.clamp(ratio, 1.0 - cfg.net_clip, 1.0 + cfg.net_clip) * adv_net[mbm]
                        net_loss = -torch.mean(torch.min(surr1, surr2))

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

                        net_loss_list.append(float(net_loss.item()))
                        vmacro_loss_list.append(float(v_loss.item()))
                        net_ratio_means.append(float(ratio.mean().item()))
                        net_ratio_stds.append(float(ratio.std(unbiased=False).item()))
                        net_kls.append(float(approx_kl.item()))
                        net_clipfracs.append(float(clipfrac.item()))
                        gn_net_list.append(float(grad_norm(net_actor)))
                        gn_vmacro_list.append(float(grad_norm(macro_critic)))

        # Net freeze diagnostics (AFTER net update)
        with torch.no_grad():
            deltas = []
            maxabs = 0.0
            for p0, p1 in zip(net_params_before, net_actor.parameters()):
                d = (p1.detach() - p0)
                deltas.append(torch.sum(d * d))
                maxabs = max(maxabs, float(torch.max(torch.abs(d)).item()))
            net_dtheta_l2 = float(torch.sqrt(torch.sum(torch.stack(deltas))).item())

        # ============================
        # Diagnostics print
        # ============================
        if it % 10 == 0:
            dbg = batch["dbg_last"]
            nd = batch.get("net_dbg_last", None) or {}

            rnet_mean = float(batch["r_net"].mean().item())
            rinfo_mean = float(batch["r_info"].mean().item())
            rpol_mean = float(batch["r_pol"].mean().item())
            rrew_mean = float(batch["r_rew"].mean().item())

            if TRAIN_GATE and gate_mu_stats:
                mu_m, mu_s = gate_mu_stats[-1]
                ls_m, ls_s = gate_ls_stats[-1]
                g_m, g_s   = gate_g_stats[-1]
                gate_ratio_m = float(np.mean(gate_ratio_means)) if gate_ratio_means else 1.0
                gate_ratio_s = float(np.mean(gate_ratio_stds))  if gate_ratio_stds  else 0.0
                gate_kl = float(np.mean(gate_kls)) if gate_kls else 0.0
                gate_clip = float(np.mean(gate_clipfracs)) if gate_clipfracs else 0.0
                gate_ent = float(np.mean(ent_gate_means)) if ent_gate_means else 0.0
                gate_loss = float(np.mean(gate_loss_list)) if gate_loss_list else 0.0
                vstep_loss = float(np.mean(vstep_loss_list)) if vstep_loss_list else 0.0
            else:
                mu_m, mu_s = (0.0, 0.0)
                ls_m, ls_s = (0.0, 0.0)
                g_m, g_s   = (float(FIX_GATE_VALUE), 0.0) if FIX_GATE else (0.0, 0.0)
                gate_ratio_m, gate_ratio_s, gate_kl, gate_clip, gate_ent = (0.0, 0.0, 0.0, 0.0, 0.0)
                gate_loss, vstep_loss = (0.0, 0.0)

            if TRAIN_NET:
                net_ratio_m = float(np.mean(net_ratio_means)) if net_ratio_means else 1.0
                net_ratio_s = float(np.mean(net_ratio_stds))  if net_ratio_stds  else 0.0
                net_kl = float(np.mean(net_kls)) if net_kls else 0.0
                net_clip = float(np.mean(net_clipfracs)) if net_clipfracs else 0.0
                mean_dlogp = float(np.mean(dlogp_list)) if dlogp_list else 0.0
                net_loss = float(np.mean(net_loss_list)) if net_loss_list else 0.0
                vmacro_loss = float(np.mean(vmacro_loss_list)) if vmacro_loss_list else 0.0
            else:
                net_ratio_m, net_ratio_s, net_kl, net_clip, mean_dlogp = (0.0, 0.0, 0.0, 0.0, 0.0)
                net_loss, vmacro_loss = (0.0, 0.0)

            print(
                f"iter={it:04d} | "
                f"r_reg={float(batch['r_reg'].mean().item()):+.4f}±{float(batch['r_reg'].std(unbiased=False).item()):.4f} | "
                f"r_net={rnet_mean:+.4f} (info={rinfo_mean:+.4f}, pol={rpol_mean:+.4f}, rew={rrew_mean:+.4f}) | "
                f"GATE(TRAIN={TRAIN_GATE}, FIX={FIX_GATE}): loss={gate_loss:+.3f} V={vstep_loss:.3f} "
                f"ratio={gate_ratio_m:.3f}±{gate_ratio_s:.3f} kl={gate_kl:+.4f} clip={gate_clip:.3f} ent={gate_ent:+.3f} | "
                f"NET(TRAIN={TRAIN_NET}, FIX={FIX_NET}, M={M}): loss={net_loss:+.3f} V={vmacro_loss:.3f} "
                f"ratio={net_ratio_m:.3f}±{net_ratio_s:.3f} kl={net_kl:+.4f} clip={net_clip:.3f} dlogp={mean_dlogp:.4f} | "
                f"NET_dtheta_l2={net_dtheta_l2:.3e} NET_dtheta_max={maxabs:.3e} | "
                f"logits_mu={nd.get('logits_mean',0.0):+.3f} logits_std={nd.get('logits_std',0.0):.3f} "
                f"ent={nd.get('ent_mean',0.0):.3f}±{nd.get('ent_std',0.0):.3f} | "
                f"gate_mu={mu_m:+.3f}±{mu_s:.3f} gate_logstd={ls_m:+.3f}±{ls_s:.3f} gate={g_m:.3f}±{g_s:.3f} | "
                f"dbg[riskS]={dbg['riskS']:.3g} dbg[flow2]={dbg['flow2']:.3g} dbg[gini]={dbg['gini']:.3g} "
                f"dbg[dP]={dbg['dP']:.3g} dbg[g_mean]={dbg['g_mean']:.3g}"
            )


if __name__ == "__main__":
    train_v10_7()
