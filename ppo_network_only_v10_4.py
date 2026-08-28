# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 13:39:33 2025

@author: hg2e25
"""

# -*- coding: utf-8 -*-
"""
v10.4 — CLEAN, ONE-PIECE FILE (based on your v10.3)
Applies the 4 requested fixes:

(A2) Soft constraint on average gate:
     r_gate = - w_gate * ( mean(g) - g_target )^2

(B) Split actor advantages / PPO losses into TWO parts:
     - Gate PPO update every step (ratio_gate)
     - Network PPO update only when netmask==1 (ratio_net)
   Uses separate (normalized) advantages:
     adv_gate : normalized over all timesteps
     adv_net  : normalized over net timesteps only (0 elsewhere)

(C) Critic stabilization beyond reward caps:
     - Return/target clipping for critic (ret_clip)
     - Huber (SmoothL1) loss instead of MSE

(D) Stronger info-task than mean(b):
     Uses in-degree-weighted aggregate belief:
        b_agg = sum_i w_i * b_i  where w_i ∝ (in_deg_i + eps)
     pred_err = (b_agg - R)^2   (then capped as before)

Notes:
- Gate is stochastic squashed Gaussian; logprob included in PPO ratio via ratio_gate.
- Network sampling uses masked sequential sampling with global indegree cap.
- Network weights are deterministic softmax(w_logits) (stable).
- Logprob scale remains mean-normalized:
      logp_net  /(N*K)   already in sampler/evaluator
      logp_gate /N       already mean over agents
- We DO NOT mix net and gate into one logp anymore (this is the key B-fix).

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


def normalize_adv(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)


# =============================================================================
# Environment
# =============================================================================

class InfoNetworkBondEnvV10_4:
    """
    Same env as v10.3, but with:
      - (D) stronger info-task: in-degree weighted aggregate belief
      - (A2) soft constraint on average gate (mean(g) -> g_target)
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
        # latent / signals / beliefs
        rho_y=0.98,
        sigma_y=0.02,
        sigma_s=0.05,
        omega_social=0.7,
        sigma_belief=0.02,
        # risk
        beta_risk=None,
        # reward weights
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=0.5,
        w_net=0.5,
        w_info=2.0,
        # (A2) gate constraint
        w_gate=0.2,
        g_target=0.4,
        # reward stabilization caps
        cap_riskS=50.0,
        cap_flow4=5.0,
        cap_pred=2.0,
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
        self.w_info = float(w_info)

        # (A2)
        self.w_gate = float(w_gate)
        self.g_target = float(g_target)

        self.cap_riskS = cap_riskS
        self.cap_flow4 = cap_flow4
        self.cap_pred = cap_pred
        self.cap_reward = cap_reward

        self.rng = np.random.default_rng(seed)

        # runtime set by trainer
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

    def _build_obs(self):
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)

        Pb = self.P @ self.b
        var_b = float(np.var(self.b))
        vol20 = float(np.sqrt(max(self.risk_v, 0.0)))
        mean_abs_deltax = float(np.mean(np.abs(self.delta_x_prev)))

        obs = []
        for i in range(self.N):
            o_i = np.array([
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
            ], dtype=float)
            obs.append(o_i)
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

        return self._build_obs()

    def step(self, g_gate, neighbors=None, w=None):
        # Apply network action (slow timescale)
        if (self.t % self.net_period) == 0 and neighbors is not None and w is not None:
            self.P_prev = self.P.copy()
            self.P = row_stochastic_from_neighbors(self.N, neighbors, w)

        # Latent evolves
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        # Belief update
        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        # Financial action (rule-based w/ learned gate)
        g_gate = np.clip(np.asarray(g_gate, dtype=float), 0.0, 1.0)
        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * g_gate * signal)

        delta_x = a_fin * self.x_max
        tc = self.tau * np.abs(delta_x)
        self.x = self.x + delta_x

        # Price formation
        net_flow = float(np.sum(delta_x))
        eps_p = float(self.rng.normal(0.0, self.sigma_eps))

        self.p_prev = self.p
        self.p = self.p + self.kappa * net_flow + eps_p
        self.p = max(self.p, 1e-6)

        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        # Risk (EWMA R^2)
        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        # Cash update
        self.c = self.c - self.p * delta_x - tc
        self.delta_x_prev = delta_x.copy()

        # -----------------------------
        # Reward shaping
        # -----------------------------
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        dP = float(np.mean(np.abs(self.P - self.P_prev)))

        flow2 = float(np.mean(delta_x ** 2))
        flow4 = float(np.mean(delta_x ** 4))

        riskS = float(self.risk_v / max(self.risk_unit, 1e-18))

        # (D) stronger info-task: in-degree weighted aggregate belief
        w_deg = (in_deg + 1e-12)
        w_deg = w_deg / (w_deg.sum() + 1e-12)
        b_agg = float(np.sum(w_deg * self.b))
        pred_err = float((b_agg - R) ** 2)

        # (A2) soft gate constraint
        g_mean = float(np.mean(g_gate))
        gate_dev2 = float((g_mean - self.g_target) ** 2)

        # --- caps to prevent outliers wrecking critic ---
        riskS_c = clip_value(riskS, self.cap_riskS)
        flow4_c = float(np.clip(flow4, 0.0, self.cap_flow4)) if self.cap_flow4 else flow4
        pred_c  = float(np.clip(pred_err, 0.0, self.cap_pred)) if self.cap_pred else pred_err

        r = 0.0
        r -= self.w_risk * riskS_c
        r -= self.w_flow2 * flow2
        r -= self.w_flow4 * flow4_c
        r -= self.w_gini * gini_in
        r -= self.w_net * dP
        r -= self.w_info * pred_c
        r -= self.w_gate * gate_dev2

        # optional total reward clip
        if self.cap_reward is not None and self.cap_reward > 0:
            r = float(np.clip(r, -self.cap_reward, self.cap_reward))

        self.t += 1
        done = self.t >= self.horizon

        info = {
            "risk_v": float(self.risk_v),
            "riskS": float(riskS),
            "riskS_c": float(riskS_c),
            "flow2": float(flow2),
            "flow4": float(flow4),
            "flow4_c": float(flow4_c),
            "gini": float(gini_in),
            "dP": float(dP),
            "b_agg": float(b_agg),
            "pred_err": float(pred_err),
            "pred_c": float(pred_c),
            "g_mean": float(g_mean),
            "gate_dev2": float(gate_dev2),
            "R": float(R),
            "price": float(self.p),
        }
        return self._build_obs(), float(r), done, info


# =============================================================================
# Policy nets
# =============================================================================

LOG_2PI = float(math.log(2.0 * math.pi))


class ActorNet(nn.Module):
    """
    Produces:
      - neighbor logits: (N,N)
      - weight logits:   (N,K)
      - gate params: mu (N,), logstd (N,) with smooth bounded mapping
    """

    def __init__(self, obs_dim: int, hidden: int = 128, emb: int = 64, K: int = 5):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.to_emb = nn.Linear(hidden, emb)
        self.to_wlogits = nn.Linear(hidden, K)

        self.to_gate_mu = nn.Linear(hidden, 1)
        self.to_gate_logstd_raw = nn.Linear(hidden, 1)

    def forward(self, obs_t: torch.Tensor, logstd_min: float, logstd_max: float):
        h = self.mlp(obs_t)

        e = self.to_emb(h)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        scale = 1.0 / math.sqrt(e.shape[-1])
        logits = (e @ e.t()) * scale

        w_logits = self.to_wlogits(h)

        mu = self.to_gate_mu(h).squeeze(-1)              # (N,)
        logstd_raw = self.to_gate_logstd_raw(h).squeeze(-1)  # (N,)

        s = torch.sigmoid(logstd_raw)
        gate_logstd = logstd_min + (logstd_max - logstd_min) * s

        return logits, w_logits, mu, gate_logstd


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        pooled = obs_t.mean(dim=0, keepdim=True)
        return self.v(pooled).squeeze(-1)


# =============================================================================
# Network sampling with indegree cap
# =============================================================================

@torch.no_grad()
def masked_sequential_sample_neighbors(
    logits: torch.Tensor,
    K: int,
    indeg_cap: int,
):
    """
    Samples neighbors (N,K) without replacement per row with a global indegree cap.
    Returns:
      neighbors: (N,K)
      logp_net_norm: scalar normalized by (N*K)
    """
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
                j = idx[torch.randint(0, idx.numel(), (1,), device=device)].item()
                neighbors[i, k] = j
                logp_total = logp_total + torch.log(torch.tensor(1e-12, device=device))
                indeg[j] += 1
                p[j] = 0.0
                continue

            p = p / s
            cat = torch.distributions.Categorical(p)
            j = int(cat.sample().item())
            neighbors[i, k] = j
            logp_total = logp_total + torch.log(p[j] + 1e-12)

            indeg[j] += 1
            p[j] = 0.0

    return neighbors, logp_total / float(N * K)


def evaluate_logprob_neighbors_masked(
    logits: torch.Tensor,
    neighbors: torch.Tensor,
    K: int,
    indeg_cap: int,
):
    """
    Recomputes normalized logprob ( / (N*K) ) for already chosen neighbors,
    mirroring the same sequential masking logic.
    """
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

    return logp_total / float(N * K)


# =============================================================================
# Squashed Gaussian gate
# =============================================================================

def gaussian_logprob(x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * logstd)
    return -0.5 * ((x - mu) ** 2 / (var + 1e-8) + 2.0 * logstd + LOG_2PI)


def sample_squashed_gaussian_gate(mu: torch.Tensor, logstd: torch.Tensor):
    """
    Returns:
      gate: (N,) in (0,1)
      logp_gate_norm: scalar mean over agents
      ent_gate_mean: scalar mean entropy of underlying Normal (diagnostic)
    """
    std = torch.exp(logstd)
    eps = torch.randn_like(mu)
    z = mu + std * eps
    u = torch.tanh(z)
    gate = 0.5 * (u + 1.0)

    logp_z = gaussian_logprob(z, mu, logstd)          # (N,)
    log_det = torch.log(1.0 - u * u + 1e-6)           # (N,)
    logp_u = logp_z - log_det
    logp_g = logp_u + math.log(2.0)                   # dg/du=0.5 => +log2
    logp_gate_norm = logp_g.mean()

    ent = 0.5 * (LOG_2PI + 1.0) + logstd              # (N,)
    ent_gate_mean = ent.mean()

    return gate, logp_gate_norm, ent_gate_mean


def logprob_squashed_gaussian_gate(gate: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
    g = torch.clamp(gate, 1e-6, 1.0 - 1e-6)
    u = 2.0 * g - 1.0
    u = torch.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6)

    z = 0.5 * (torch.log1p(u) - torch.log1p(-u))      # atanh(u)

    logp_z = gaussian_logprob(z, mu, logstd)          # (N,)
    log_det = torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6)
    logp_u = logp_z - log_det
    logp_g = logp_u + math.log(2.0)

    return logp_g.mean()


# =============================================================================
# PPO
# =============================================================================

@dataclass
class PPOConfig:
    # env
    horizon: int = 1000
    gamma_fin: float = 5.0
    net_period: int = 5
    K: int = 5
    indeg_cap: int = 8

    # ppo
    n_iters: int = 200
    rollout_len: int = 200
    mini_epochs: int = 2
    minibatch: int = 64

    lr_actor: float = 5e-5
    lr_critic: float = 5e-4

    clip_range: float = 0.1
    gae_lambda: float = 0.95
    discount: float = 0.99

    vf_coef: float = 1.0
    max_grad_norm: float = 1.0

    # critic stabilization (C)
    ret_clip: float = 20.0          # clip critic targets
    huber_delta: float = 1.0        # SmoothL1 beta in PyTorch is 1.0 by default

    # gate
    logstd_min: float = -2.5
    logstd_max: float = -0.5
    ent_coef_gate: float = 0.005    # mean-scaled

    # (B) weights for split losses
    w_pg_gate: float = 1.0
    w_pg_net: float = 1.0

    device: str = "cpu"


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


def rollout_one(env: InfoNetworkBondEnvV10_4, actor: ActorNet, critic: CriticNet, cfg: PPOConfig):
    device = cfg.device

    obs = env.reset()
    obs_dim = len(obs[0])

    obs_list = []
    v_list = []
    r_list = []
    done_list = []

    netmask_list = []
    neigh_list = []
    w_list = []
    logp_net_list = []

    gate_list = []
    logp_gate_list = []
    ent_gate_list = []

    dbg_last = {}

    for _ in range(cfg.rollout_len):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)  # (N,obs_dim)
        v_t = critic(obs_t)

        logits, w_logits, gate_mu, gate_logstd = actor(obs_t, cfg.logstd_min, cfg.logstd_max)
        gate, logp_gate_norm, ent_gate_mean = sample_squashed_gaussian_gate(gate_mu, gate_logstd)

        do_net = (env.t % cfg.net_period) == 0
        if do_net:
            neighbors, logp_net_norm = masked_sequential_sample_neighbors(
                logits=logits, K=cfg.K, indeg_cap=cfg.indeg_cap
            )
            w = torch.softmax(w_logits, dim=-1)
        else:
            neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
            logp_net_norm = torch.zeros((), device=device)
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)

        if do_net:
            obs, r, done, info = env.step(
                g_gate=gate.detach().cpu().numpy(),
                neighbors=neighbors.detach().cpu().numpy(),
                w=w.detach().cpu().numpy(),
            )
        else:
            obs, r, done, info = env.step(
                g_gate=gate.detach().cpu().numpy(),
                neighbors=None,
                w=None,
            )

        obs_list.append(obs_t)
        v_list.append(v_t.squeeze())
        r_list.append(float(r))
        done_list.append(float(done))

        netmask_list.append(float(do_net))
        neigh_list.append(neighbors)
        w_list.append(w)
        logp_net_list.append(logp_net_norm)

        gate_list.append(gate)
        logp_gate_list.append(logp_gate_norm)
        ent_gate_list.append(ent_gate_mean)

        dbg_last = info
        if done:
            break

    return {
        "obs": torch.stack(obs_list),  # (T,N,obs_dim)
        "v": torch.stack(v_list),      # (T,)
        "rew": torch.tensor(r_list, dtype=torch.float32, device=device),
        "done": torch.tensor(done_list, dtype=torch.float32, device=device),
        "netmask": torch.tensor(netmask_list, dtype=torch.float32, device=device),
        "neighbors": neigh_list,
        "w": w_list,
        "logp_net": torch.stack(logp_net_list),     # (T,) mean-normalized
        "gate": torch.stack(gate_list),             # (T,N)
        "logp_gate": torch.stack(logp_gate_list),   # (T,) mean-normalized
        "ent_gate": torch.stack(ent_gate_list),     # (T,)
        "dbg_last": dbg_last,
        "obs_dim": obs_dim,
    }


def train_v10_4():
    # -----------------------------
    # Config
    # -----------------------------
    cfg = PPOConfig(
        horizon=1000,
        gamma_fin=5.0,
        K=5,
        indeg_cap=8,
        net_period=5,
        n_iters=200,
        rollout_len=200,
        mini_epochs=2,
        minibatch=64,
        lr_actor=5e-5,
        lr_critic=5e-4,
        clip_range=0.1,
        discount=0.99,
        gae_lambda=0.95,
        vf_coef=1.0,
        max_grad_norm=1.0,
        ret_clip=20.0,
        huber_delta=1.0,
        logstd_min=-2.5,
        logstd_max=-0.5,
        ent_coef_gate=0.005,
        w_pg_gate=1.0,
        w_pg_net=1.0,
        device="cpu",
    )

    env_kwargs = dict(
        N=50,
        K=cfg.K,
        net_period=cfg.net_period,
        indeg_cap=cfg.indeg_cap,
        horizon=cfg.horizon,
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
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=0.5,
        w_net=0.5,
        w_info=2.0,
        # (A2)
        w_gate=0.2,
        g_target=0.4,
        # stabilization caps
        cap_riskS=50.0,
        cap_flow4=5.0,
        cap_pred=2.0,
        cap_reward=50.0,
    )

    print("=" * 118)
    print(f"CONFIG v10.4: horizon={cfg.horizon} | gamma_fin={cfg.gamma_fin} | K={cfg.K} | indeg_cap={cfg.indeg_cap} | net_period={cfg.net_period}")
    print(f"Reward: w_risk={env_kwargs['w_risk']} w_flow2={env_kwargs['w_flow2']} w_flow4={env_kwargs['w_flow4']} "
          f"w_gini={env_kwargs['w_gini']} w_net={env_kwargs['w_net']} w_info={env_kwargs['w_info']} "
          f"w_gate={env_kwargs['w_gate']} g_target={env_kwargs['g_target']}")
    print(f"PPO: lr_actor={cfg.lr_actor} clip_range={cfg.clip_range} mini_epochs={cfg.mini_epochs} minibatch={cfg.minibatch}")
    print(f"Split PG: w_pg_gate={cfg.w_pg_gate} (every step), w_pg_net={cfg.w_pg_net} (net steps only).")
    print(f"Gate: squashed Gaussian, logstd in [{cfg.logstd_min}, {cfg.logstd_max}], entropy bonus gate-only={cfg.ent_coef_gate} (mean-scaled)")
    print(f"Critic: Huber + ret_clip={cfg.ret_clip}")
    print(f"Info-task: in-degree weighted belief aggregate (stronger than mean(b))")
    print("=" * 118)

    # -----------------------------
    # Init
    # -----------------------------
    seed_train = 0
    set_seed(seed_train)

    env = InfoNetworkBondEnvV10_4(seed=seed_train, **env_kwargs)
    env.gamma_fin = cfg.gamma_fin

    obs0 = env.reset()
    obs_dim = len(obs0[0])

    actor = ActorNet(obs_dim=obs_dim, hidden=128, emb=64, K=cfg.K).to(cfg.device)
    critic = CriticNet(obs_dim=obs_dim, hidden=128).to(cfg.device)

    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    huber = nn.SmoothL1Loss(beta=cfg.huber_delta)

    # -----------------------------
    # Train loop
    # -----------------------------
    for it in range(cfg.n_iters):
        batch = rollout_one(env, actor, critic, cfg)

        obs = batch["obs"]                 # (T,N,obs_dim)
        T = obs.shape[0]
        netmask = batch["netmask"].detach()  # (T,)

        v = batch["v"].detach()
        with torch.no_grad():
            v_next = critic(obs[-1]) * (1.0 - batch["done"][-1])

        adv_raw, ret_raw = compute_gae(
            rew=batch["rew"],
            done=batch["done"],
            v=v,
            v_next=v_next,
            discount=cfg.discount,
            lam=cfg.gae_lambda,
        )

        # (C) critic target clipping
        ret = torch.clamp(ret_raw, -cfg.ret_clip, cfg.ret_clip)

        # (B) split advantages: all-steps vs net-steps-only (normalized separately)
        adv_gate = normalize_adv(adv_raw.detach())

        adv_net = torch.zeros_like(adv_gate)
        net_idx = (netmask > 0.5)
        if net_idx.any():
            adv_net_vals = adv_raw.detach()[net_idx]
            adv_net_vals = normalize_adv(adv_net_vals)
            adv_net[net_idx] = adv_net_vals
        # if no net steps (shouldn't happen), adv_net stays zero

        # old logps (mean-normalized)
        logp_gate_old = batch["logp_gate"].detach()      # (T,)
        logp_net_old = batch["logp_net"].detach()        # (T,)
        gate_old = batch["gate"].detach()                # (T,N)

        idx = torch.arange(T, device=cfg.device)

        # diagnostics
        actor_losses = []
        critic_losses = []

        ratio_gate_means, ratio_gate_stds = [], []
        ratio_net_means, ratio_net_stds = [], []

        approx_kl_gate, approx_kl_net = [], []
        clipfrac_gate, clipfrac_net = [], []

        ent_gate_means = []
        gn_actor_list = []
        gn_critic_list = []

        gate_mu_stats = []
        gate_logstd_stats = []
        gate_stats = []

        for _ in range(cfg.mini_epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                if mb.numel() == 0:
                    continue

                logp_gate_new_list = []
                logp_net_new_list = []
                ent_gate_list = []

                v_mb = torch.zeros((mb.numel(),), device=cfg.device)

                mu_list = []
                logstd_list = []
                g_list_mb = []

                for j, t in enumerate(mb.tolist()):
                    obs_t = obs[t]
                    v_mb[j] = critic(obs_t)

                    logits_t, w_logits_t, mu_t, logstd_t = actor(obs_t, cfg.logstd_min, cfg.logstd_max)

                    # gate logp always
                    lp_gate = logprob_squashed_gaussian_gate(gate_old[t], mu_t, logstd_t)
                    logp_gate_new_list.append(lp_gate)

                    # net logp only if net-step
                    if netmask[t] > 0.5:
                        neigh_t = batch["neighbors"][t]
                        lp_net = evaluate_logprob_neighbors_masked(
                            logits=logits_t,
                            neighbors=neigh_t,
                            K=cfg.K,
                            indeg_cap=cfg.indeg_cap,
                        )
                    else:
                        lp_net = torch.zeros((), device=cfg.device)
                    logp_net_new_list.append(lp_net)

                    ent_gate = (0.5 * (LOG_2PI + 1.0) + logstd_t).mean()
                    ent_gate_list.append(ent_gate)

                    mu_list.append(mu_t.detach())
                    logstd_list.append(logstd_t.detach())
                    g_list_mb.append(gate_old[t].detach())

                logp_gate_new = torch.stack(logp_gate_new_list)  # (B,)
                logp_net_new = torch.stack(logp_net_new_list)    # (B,)
                ent_gate = torch.stack(ent_gate_list)            # (B,)

                # PPO ratios
                logp_gate_old_mb = logp_gate_old[mb]
                logp_net_old_mb = logp_net_old[mb]

                ratio_gate = torch.exp(logp_gate_new - logp_gate_old_mb)
                ratio_net = torch.exp(logp_net_new - logp_net_old_mb)

                # advantages
                adv_gate_mb = adv_gate[mb]
                adv_net_mb = adv_net[mb]
                ret_mb = ret[mb]

                # (B) separate PPO losses
                s1g = ratio_gate * adv_gate_mb
                s2g = torch.clamp(ratio_gate, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_gate_mb
                policy_loss_gate = -torch.mean(torch.min(s1g, s2g))

                # net loss only where netmask==1 (within minibatch)
                m = (netmask[mb] > 0.5).float()
                if m.sum() > 0:
                    s1n = ratio_net * adv_net_mb
                    s2n = torch.clamp(ratio_net, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_net_mb
                    policy_loss_net = -(m * torch.min(s1n, s2n)).sum() / (m.sum() + 1e-8)
                else:
                    policy_loss_net = torch.zeros((), device=cfg.device)

                # gate-only entropy bonus
                ent_bonus = torch.mean(ent_gate)

                actor_loss = (
                    cfg.w_pg_gate * policy_loss_gate
                    + cfg.w_pg_net * policy_loss_net
                    - cfg.ent_coef_gate * ent_bonus
                )

                # (C) Huber critic loss (with clipped targets)
                critic_loss = huber(v_mb, ret_mb)

                # update actor
                opt_a.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt_a.step()

                # update critic
                opt_c.zero_grad(set_to_none=True)
                (cfg.vf_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                opt_c.step()

                with torch.no_grad():
                    # KL + clipfrac separately
                    kl_g = torch.mean(logp_gate_old_mb - logp_gate_new)
                    cf_g = torch.mean((torch.abs(ratio_gate - 1.0) > cfg.clip_range).float())

                    if m.sum() > 0:
                        # compute on net-only subset for meaningful numbers
                        sel = (m > 0.5)
                        kl_n = torch.mean(logp_net_old_mb[sel] - logp_net_new[sel])
                        cf_n = torch.mean((torch.abs(ratio_net[sel] - 1.0) > cfg.clip_range).float())
                        rnm = float(ratio_net[sel].mean().item())
                        rns = float(ratio_net[sel].std().item())
                    else:
                        kl_n = torch.zeros((), device=cfg.device)
                        cf_n = torch.zeros((), device=cfg.device)
                        rnm = 1.0
                        rns = 0.0

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))

                ratio_gate_means.append(float(ratio_gate.mean().item()))
                ratio_gate_stds.append(float(ratio_gate.std().item()))
                approx_kl_gate.append(float(kl_g.item()))
                clipfrac_gate.append(float(cf_g.item()))

                ratio_net_means.append(rnm)
                ratio_net_stds.append(rns)
                approx_kl_net.append(float(kl_n.item()))
                clipfrac_net.append(float(cf_n.item()))

                ent_gate_means.append(float(ent_bonus.item()))
                gn_actor_list.append(grad_norm(actor))
                gn_critic_list.append(grad_norm(critic))

                # diagnostics: last minibatch stats
                mu_cat = torch.cat(mu_list, dim=0)
                ls_cat = torch.cat(logstd_list, dim=0)
                g_cat = torch.cat(g_list_mb, dim=0)

                gate_mu_stats.append((float(mu_cat.mean().item()), float(mu_cat.std().item())))
                gate_logstd_stats.append((float(ls_cat.mean().item()), float(ls_cat.std().item())))
                gate_stats.append((float(g_cat.mean().item()), float(g_cat.std().item())))

        if it % 10 == 0:
            dbg = batch["dbg_last"]
            mu_m, mu_s = gate_mu_stats[-1]
            ls_m, ls_s = gate_logstd_stats[-1]
            g_m, g_s = gate_stats[-1]

            print(
                f"iter={it:04d} | "
                f"actor_loss={np.mean(actor_losses):+.4f} | critic_loss={np.mean(critic_losses):.2f} | "
                f"R_mean={float(batch['rew'].mean().item()):+.4f} | R_std={float(batch['rew'].std().item()):.4f} | "
                f"gate_ratio={np.mean(ratio_gate_means):.3f}±{np.mean(ratio_gate_stds):.3f} "
                f"net_ratio={np.mean(ratio_net_means):.3f}±{np.mean(ratio_net_stds):.3f} | "
                f"kl_gate={np.mean(approx_kl_gate):+.4f} kl_net={np.mean(approx_kl_net):+.4f} | "
                f"clip_gate={np.mean(clipfrac_gate):.3f} clip_net={np.mean(clipfrac_net):.3f} | "
                f"ent_gate={np.mean(ent_gate_means):+.3f} | "
                f"gradA={np.mean(gn_actor_list):.3g} gradC={np.mean(gn_critic_list):.3g} | "
                f"gate_mu={mu_m:+.3f}±{mu_s:.3f} gate_logstd={ls_m:+.3f}±{ls_s:.3f} gate={g_m:.3f}±{g_s:.3f} | "
                f"dbg[riskS]={dbg['riskS']:.3g} dbg[flow2]={dbg['flow2']:.3g} dbg[gini]={dbg['gini']:.3g} dbg[dP]={dbg['dP']:.3g} "
                f"dbg[pred]={dbg['pred_err']:.3g} dbg[b_agg]={dbg['b_agg']:.3g} dbg[g_mean]={dbg['g_mean']:.3g}"
            )


if __name__ == "__main__":
    train_v10_4()
