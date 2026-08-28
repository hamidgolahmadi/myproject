# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 13:03:18 2025

@author: hg2e25
"""
# -*- coding: utf-8 -*-
"""
v10.3 — CLEAN, ONE-PIECE FILE
Applies ALL fixes discussed:

1) Gate is truly stochastic: squashed Gaussian with correct logprob (incl. tanh squash) and included in PPO ratio.
2) NO hard clamp on logstd (which killed gradients). Uses smooth mapping to [logstd_min, logstd_max].
3) Entropy bonus is gate-only AND properly scaled (mean per-agent), so it's not a huge constant.
4) Logprob scaling is normalized:
      logp_net  /= (N*K)
      logp_gate /= N
   Ratio/KL still moves, but we avoid exploding gradients.
5) Reward stabilization: clip volatile terms + optional reward clip to prevent catastrophic critic explosions.
6) PPO diagnostics: ratio mean/std, approx_kl, clipfrac, gate entropy, grad norms + gate stats.

Notes:
- Network weights w are deterministic softmax(w_logits) (stable).
- Network sampling uses masked sequential sampling with a global indegree cap.
- Policy gradient is applied only on timesteps when net action executes (t % net_period == 0) for the net part,
  but gate is updated EVERY step because gate always has a logprob.
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

class InfoNetworkBondEnvV10:
    """
    Multi-agent env:
    - Information network P (row-stochastic, each i listens to K nodes)
    - Latent fundamental y_t (AR(1))
    - Private signals s_i(t) = y_t + noise
    - Beliefs b updated via private + DeGroot diffusion
    - Price responds to trading flow (simple impact + noise)
    - PPO controls:
        (i) network rewiring every net_period (neighbors + weights)
        (ii) financial gate g_i(t) in [0,1] each step

    Reward is purely "system / network / risk" shaped:
      - riskS (EWMA of returns^2 scaled)
      - flow2, flow4 penalties
      - gini(in-degree) penalty
      - dP penalty
      - info-task pred_err penalty: (mean(b) - R)^2

    Stabilization (v10.3):
      - clip flow4 and riskS and pred_err optionally
      - optional reward clip
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
        self.b_bar_prev = float(np.mean(self.b))

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

        # aggregate belief before realized return
        self.b_bar_prev = float(np.mean(self.b))

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

        # Reward shaping
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        dP = float(np.mean(np.abs(self.P - self.P_prev)))

        flow2 = float(np.mean(delta_x ** 2))
        flow4 = float(np.mean(delta_x ** 4))

        riskS = float(self.risk_v / max(self.risk_unit, 1e-18))
        pred_err = float((self.b_bar_prev - R) ** 2)

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
            "pred_err": float(pred_err),
            "pred_c": float(pred_c),
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

        # gate head
        self.to_gate_mu = nn.Linear(hidden, 1)
        self.to_gate_logstd_raw = nn.Linear(hidden, 1)

    def forward(self, obs_t: torch.Tensor, logstd_min: float, logstd_max: float):
        """
        obs_t: (N, obs_dim)
        returns:
          logits: (N,N)
          w_logits: (N,K)
          gate_mu: (N,)
          gate_logstd: (N,) bounded smoothly
        """
        h = self.mlp(obs_t)

        e = self.to_emb(h)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        scale = 1.0 / math.sqrt(e.shape[-1])
        logits = (e @ e.t()) * scale

        w_logits = self.to_wlogits(h)

        mu = self.to_gate_mu(h).squeeze(-1)  # (N,)
        logstd_raw = self.to_gate_logstd_raw(h).squeeze(-1)  # (N,)

        # Smoothly map raw -> [logstd_min, logstd_max] without hard clamp
        # sigmoid in (0,1), then affine
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
        return self.v(pooled).squeeze(-1)  # scalar


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
                # fallback
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

    logp_net_norm = logp_total / float(N * K)
    return neighbors, logp_net_norm


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
    """
    Returns per-dimension logprob for Normal(mu, std) at x.
    x, mu, logstd shape: (N,)
    """
    var = torch.exp(2.0 * logstd)
    return -0.5 * ((x - mu) ** 2 / (var + 1e-8) + 2.0 * logstd + LOG_2PI)


def sample_squashed_gaussian_gate(
    mu: torch.Tensor,
    logstd: torch.Tensor,
):
    """
    Sample:
      z ~ N(mu, std)
      u = tanh(z) in (-1,1)
      g = 0.5 * (u + 1) in (0,1)

    Returns:
      gate: (N,) in (0,1)
      logp_gate_norm: scalar normalized by N  (mean per-agent logprob)
      ent_gate_mean: mean per-agent differential entropy of underlying Normal (diagnostic)
      z: (N,) (optional, for debugging)
    """
    std = torch.exp(logstd)
    eps = torch.randn_like(mu)
    z = mu + std * eps
    u = torch.tanh(z)

    gate = 0.5 * (u + 1.0)

    # logprob with tanh correction
    # log p(u) = log p(z) - log |det du/dz|  where du/dz = 1 - tanh(z)^2
    logp_z = gaussian_logprob(z, mu, logstd)  # (N,)
    # correction: sum log(1 - u^2); here 1D per agent, so just per agent
    log_det = torch.log(1.0 - u * u + 1e-6)   # (N,)
    logp_u = logp_z - log_det

    # scaling by 0.5 and shift doesn't change density except a constant Jacobian (|dg/du|=0.5)
    # so add -log(0.5) = +log(2)
    logp_g = logp_u + math.log(2.0)

    # normalize: mean over agents
    logp_gate_norm = logp_g.mean()

    # diagnostic entropy of underlying Normal: H = 0.5 * log(2πeσ^2)
    ent = 0.5 * (LOG_2PI + 1.0) + logstd  # (N,)
    ent_gate_mean = ent.mean()

    return gate, logp_gate_norm, ent_gate_mean, z


def logprob_squashed_gaussian_gate(
    gate: torch.Tensor,
    mu: torch.Tensor,
    logstd: torch.Tensor,
):
    """
    Compute normalized logprob (mean over agents) for a given gate in (0,1),
    under the squashed Gaussian policy.
    """
    # invert g -> u -> z
    g = torch.clamp(gate, 1e-6, 1.0 - 1e-6)
    u = 2.0 * g - 1.0
    u = torch.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6)

    # atanh(u) = 0.5*log((1+u)/(1-u))
    z = 0.5 * (torch.log1p(u) - torch.log1p(-u))

    logp_z = gaussian_logprob(z, mu, logstd)        # (N,)
    log_det = torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6)  # (N,)
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

    # gate
    logstd_min: float = -2.5
    logstd_max: float = -0.5
    ent_coef_gate: float = 0.005  # small: now entropy is properly scaled (mean)

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


def rollout_one(env: InfoNetworkBondEnvV10, actor: ActorNet, critic: CriticNet, cfg: PPOConfig):
    """
    Collect rollout and store everything needed for PPO.

    We store:
      - obs_t: (N,obs_dim)
      - value V_t
      - reward
      - done
      - netmask (1 if net action executed)
      - neighbors_t, w_t for net steps
      - logp_net_norm for net steps (else 0)
      - gate sample for every step + logp_gate_norm for every step
      - ent_gate_mean for every step (diagnostic)
    """
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

        gate, logp_gate_norm, ent_gate_mean, _ = sample_squashed_gaussian_gate(gate_mu, gate_logstd)

        do_net = (env.t % cfg.net_period) == 0
        if do_net:
            neighbors, logp_net_norm = masked_sequential_sample_neighbors(
                logits=logits, K=cfg.K, indeg_cap=cfg.indeg_cap
            )
            w = torch.softmax(w_logits, dim=-1)  # deterministic
        else:
            neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
            logp_net_norm = torch.zeros((), device=device)
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)

        # step env
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

        # store
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

    batch = {
        "obs": torch.stack(obs_list),                         # (T,N,obs_dim)
        "v": torch.stack(v_list),                             # (T,)
        "rew": torch.tensor(r_list, dtype=torch.float32, device=device),   # (T,)
        "done": torch.tensor(done_list, dtype=torch.float32, device=device),# (T,)
        "netmask": torch.tensor(netmask_list, dtype=torch.float32, device=device), # (T,)
        "neighbors": neigh_list,                              # list[T] of (N,K)
        "w": w_list,                                          # list[T] of (N,K)
        "logp_net": torch.stack(logp_net_list),               # (T,) normalized
        "gate": torch.stack(gate_list),                       # (T,N)
        "logp_gate": torch.stack(logp_gate_list),             # (T,) normalized (mean over agents)
        "ent_gate": torch.stack(ent_gate_list),               # (T,) mean entropy
        "dbg_last": dbg_last,
        "obs_dim": obs_dim,
    }
    return batch


def train_v10_3():
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
        logstd_min=-2.5,
        logstd_max=-0.5,
        ent_coef_gate=0.005,
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
        # stabilization caps
        cap_riskS=50.0,
        cap_flow4=5.0,
        cap_pred=2.0,
        cap_reward=50.0,
    )

    print("=" * 110)
    print(f"CONFIG v10.3: horizon={cfg.horizon} | gamma_fin={cfg.gamma_fin} | K={cfg.K} | indeg_cap={cfg.indeg_cap} | net_period={cfg.net_period}")
    print(f"Reward: w_risk={env_kwargs['w_risk']} w_flow2={env_kwargs['w_flow2']} w_flow4={env_kwargs['w_flow4']} w_gini={env_kwargs['w_gini']} w_net={env_kwargs['w_net']} w_info={env_kwargs['w_info']}")
    print(f"PPO: lr_actor={cfg.lr_actor} clip_range={cfg.clip_range} mini_epochs={cfg.mini_epochs} | Gate logstd in [{cfg.logstd_min}, {cfg.logstd_max}] | Entropy bonus: gate-only (mean-scaled)")
    print("LOGPROB SCALE: logp_net/(N*K) + logp_gate/N (both are mean-normalized) -> stable ratio/KL.")
    print("Reward caps: riskS, flow4, pred_err, total reward (prevents critic explosions).")
    print("=" * 110)

    # -----------------------------
    # Init
    # -----------------------------
    seed_train = 0
    set_seed(seed_train)

    env = InfoNetworkBondEnvV10(seed=seed_train, **env_kwargs)
    env.gamma_fin = cfg.gamma_fin

    obs0 = env.reset()
    obs_dim = len(obs0[0])

    actor = ActorNet(obs_dim=obs_dim, hidden=128, emb=64, K=cfg.K).to(cfg.device)
    critic = CriticNet(obs_dim=obs_dim, hidden=128).to(cfg.device)

    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # -----------------------------
    # Train loop
    # -----------------------------
    for it in range(cfg.n_iters):
        batch = rollout_one(env, actor, critic, cfg)

        obs = batch["obs"]     # (T,N,obs_dim)
        T = obs.shape[0]

        v = batch["v"].detach()  # (T,)
        with torch.no_grad():
            v_next = critic(obs[-1]) * (1.0 - batch["done"][-1])

        adv, ret = compute_gae(
            rew=batch["rew"],
            done=batch["done"],
            v=v,
            v_next=v_next,
            discount=cfg.discount,
            lam=cfg.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # old logp (normalized)
        logp_old = (batch["logp_net"] + batch["logp_gate"]).detach()  # (T,)
        gate_old = batch["gate"].detach()                              # (T,N)
        netmask = batch["netmask"].detach()                            # (T,)

        idx = torch.arange(T, device=cfg.device)

        # diagnostics aggregators
        actor_losses = []
        critic_losses = []
        approx_kls = []
        clipfracs = []
        ratio_means = []
        ratio_stds = []
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

                # recompute current logp for minibatch
                logp_new_list = []
                ent_gate_list = []
                v_mb = torch.zeros((mb.numel(),), device=cfg.device)

                mu_list = []
                logstd_list = []
                gate_list_mb = []

                for j, t in enumerate(mb.tolist()):
                    obs_t = obs[t]  # (N,obs_dim)
                    v_mb[j] = critic(obs_t)

                    logits_t, w_logits_t, mu_t, logstd_t = actor(obs_t, cfg.logstd_min, cfg.logstd_max)

                    # gate logp (mean-normalized)
                    lp_gate = logprob_squashed_gaussian_gate(gate_old[t], mu_t, logstd_t)

                    # net logp only if net action executed at that timestep
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

                    logp_new = lp_net + lp_gate
                    logp_new_list.append(logp_new)

                    # gate entropy diagnostic (mean per agent)
                    ent_gate = (0.5 * (LOG_2PI + 1.0) + logstd_t).mean()
                    ent_gate_list.append(ent_gate)

                    mu_list.append(mu_t.detach())
                    logstd_list.append(logstd_t.detach())
                    gate_list_mb.append(gate_old[t].detach())

                logp_new = torch.stack(logp_new_list)      # (B,)
                ent_gate = torch.stack(ent_gate_list)      # (B,)

                # ratio + PPO objective
                logp_old_mb = logp_old[mb]
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                ratio = torch.exp(logp_new - logp_old_mb)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Gate-only entropy bonus (mean-scaled)
                ent_bonus = torch.mean(ent_gate)
                actor_loss = policy_loss - cfg.ent_coef_gate * ent_bonus

                critic_loss = torch.mean((v_mb - ret_mb) ** 2)

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
                    approx_kl = torch.mean(logp_old_mb - logp_new)
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_range).float())

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                approx_kls.append(float(approx_kl.item()))
                clipfracs.append(float(clipfrac.item()))
                ratio_means.append(float(ratio.mean().item()))
                ratio_stds.append(float(ratio.std().item()))
                ent_gate_means.append(float(ent_bonus.item()))

                gn_actor_list.append(grad_norm(actor))
                gn_critic_list.append(grad_norm(critic))

                # gate stats (diagnostic)
                mu_cat = torch.cat(mu_list, dim=0)
                ls_cat = torch.cat(logstd_list, dim=0)
                g_cat = torch.cat(gate_list_mb, dim=0)

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
                f"ratio={np.mean(ratio_means):.3f}±{np.mean(ratio_stds):.3f} | "
                f"kl={np.mean(approx_kls):+.4f} | clipfrac={np.mean(clipfracs):.3f} | "
                f"ent_gate={np.mean(ent_gate_means):.3f} | "
                f"gradA={np.mean(gn_actor_list):.3g} gradC={np.mean(gn_critic_list):.3g} | "
                f"gate_mu={mu_m:+.3f}±{mu_s:.3f} gate_logstd={ls_m:+.3f}±{ls_s:.3f} gate={g_m:.3f}±{g_s:.3f} | "
                f"dbg[riskS]={dbg['riskS']:.3g} dbg[flow2]={dbg['flow2']:.3g} dbg[gini]={dbg['gini']:.3g} dbg[dP]={dbg['dP']:.3g} dbg[pred]={dbg['pred_err']:.3g}"
            )


if __name__ == "__main__":
    train_v10_3()
