# -*- coding: utf-8 -*-
"""
ppo_network_only_v10.py

v10: Info-task reward + Path-B gate is STOCHASTIC (Gaussian -> sigmoid squash)
Fixes:
- NameError from wrong type-hint class name
- Adds: from __future__ import annotations (prevents forward-ref issues)

Key changes vs v9:
- Gate is a real PPO action:
    u ~ Normal(mu, std)
    g = sigmoid(u)  in (0,1)
  We store u and logp_old, and in PPO we recompute logp_new for the SAME u.
- PPO ratio uses: logp_total = logp_gate + netmask * logp_net
- Diagnostics printed: ratio mean/std, approx_kl, clipfrac, entropy, grad norms
"""

from __future__ import annotations

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


# =============================================================================
# Environment
# =============================================================================

class InfoNetworkBondEnvV10:
    """
    Multi-agent info network with:
    - Row-stochastic P, each i listens to K nodes.
    - Latent fundamental y_t, private signals s_i(t)
    - Belief mixing: b(t+1) = (1-omega) s + omega (P b) + noise
    - PPO controls:
        (i) network rewiring every net_period
        (ii) financial gate g_i(t) ∈ (0,1) each step (now stochastic in policy)
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
        # latent fundamental / private signal
        rho_y=0.98,
        sigma_y=0.02,
        sigma_s=0.05,
        omega_social=0.7,
        sigma_belief=0.02,
        # risk (EWMA of R^2)
        beta_risk=None,
        # reward weights
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=0.5,
        w_net=0.5,
        w_info=2.0,
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

        self.rng = np.random.default_rng(seed)

        # runtime parameter set by runner
        self.gamma_fin = 5.0

        self.reset()

    def _init_P_random_topk(self):
        P = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            candidates = [j for j in range(self.N) if j != i]
            nbrs = self.rng.choice(candidates, size=self.K, replace=False)
            w = self.rng.random(self.K)
            w = w / (w.sum() + 1e-12)
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

        self.b_bar_prev = float(np.mean(self.b))
        return self._build_obs()

    def step(self, g_gate, neighbors=None, w=None):
        # Apply network action
        if (self.t % self.net_period) == 0 and neighbors is not None and w is not None:
            self.P_prev = self.P.copy()
            self.P = row_stochastic_from_neighbors(self.N, neighbors, w)

        # Fundamental evolves
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        # Belief update
        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        # Store aggregate belief BEFORE price realization (prediction target)
        self.b_bar_prev = float(np.mean(self.b))

        # Financial action (rule-based with gate)
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

        # Risk EWMA
        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        # Cash update
        self.c = self.c - self.p * delta_x - tc
        self.delta_x_prev = delta_x.copy()

        # Reward shaping terms
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)
        dP = float(np.mean(np.abs(self.P - self.P_prev)))
        flow2 = float(np.mean(delta_x ** 2))
        flow4 = float(np.mean(delta_x ** 4))
        riskS = float(self.risk_v / max(self.risk_unit, 1e-18))
        pred_err = float((self.b_bar_prev - R) ** 2)

        r = 0.0
        r -= self.w_risk * riskS
        r -= self.w_flow2 * flow2
        r -= self.w_flow4 * flow4
        r -= self.w_gini * gini_in
        r -= self.w_net * dP
        r -= self.w_info * pred_err

        self.t += 1
        done = self.t >= self.horizon

        info = {
            "risk_v": float(self.risk_v),
            "riskS": riskS,
            "flow2": flow2,
            "flow4": flow4,
            "gini": gini_in,
            "dP": dP,
            "pred_err": pred_err,
            "R": float(R),
            "price": float(self.p),
        }
        return self._build_obs(), float(r), done, info


# =============================================================================
# Actor / Critic
# =============================================================================

class ActorNetV10(nn.Module):
    """
    Outputs:
      - logits_ij (N,N) for neighbor sampling
      - w_logits (N,K) deterministic softmax weights
      - gate Gaussian params: mu (N,), logstd (N,)
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
        self.to_gate_logstd = nn.Linear(hidden, 1)

    def forward(self, obs_t: torch.Tensor):
        h = self.mlp(obs_t)
        e = self.to_emb(h)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)

        scale = 1.0 / math.sqrt(e.shape[-1])
        logits = (e @ e.t()) * scale

        w_logits = self.to_wlogits(h)

        gate_mu = self.to_gate_mu(h).squeeze(-1)         # (N,)
        gate_logstd = self.to_gate_logstd(h).squeeze(-1) # (N,)

        # clamp for stability
        gate_logstd = torch.clamp(gate_logstd, -5.0, 2.0)

        return logits, w_logits, gate_mu, gate_logstd


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
        return self.v(pooled).squeeze(-1)  # scalar tensor


# =============================================================================
# Gate: squashed Gaussian logprob
# =============================================================================

LOG2PI = math.log(2.0 * math.pi)


def normal_logprob(u: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    # elementwise log N(u|mu,std)
    var = torch.exp(2.0 * logstd)
    return -0.5 * (((u - mu) ** 2) / (var + 1e-12) + 2.0 * logstd + LOG2PI)


def gate_sample_and_logprob(mu: torch.Tensor, logstd: torch.Tensor):
    """
    Sample u ~ N(mu,std); g = sigmoid(u)
    Return:
      g: (N,) in (0,1)
      u: (N,) pre-squash
      logp_g: scalar (mean over agents)
      ent_u: scalar (mean entropy of Normal)  [diagnostic only]
    """
    std = torch.exp(logstd)
    eps = torch.randn_like(mu)
    u = mu + std * eps
    g = torch.sigmoid(u)

    # logp(u)
    logp_u = normal_logprob(u, mu, logstd)

    # change-of-variables for sigmoid:
    # g = sigmoid(u) => du/dg = 1/(g(1-g)) => log|du/dg| = -log(g(1-g))
    # so logp(g) = logp(u) - log|du/dg| = logp(u) + log(g(1-g))
    log_j = torch.log(g * (1.0 - g) + 1e-12)
    logp_g = (logp_u + log_j).mean()  # mean over agents

    # Normal entropy (diag) per element: 0.5*(1+log(2πσ^2))
    ent_u = (0.5 * (1.0 + LOG2PI + 2.0 * logstd)).mean()

    return g, u, logp_g, ent_u


def gate_logprob_from_u(u: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
    """
    Recompute logp(g) for the SAME stored u.
    Return scalar mean over agents.
    """
    g = torch.sigmoid(u)
    logp_u = normal_logprob(u, mu, logstd)
    log_j = torch.log(g * (1.0 - g) + 1e-12)
    return (logp_u + log_j).mean()


# =============================================================================
# Masked sequential sampling with indegree cap (network action)
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
            cat = torch.distributions.Categorical(p)
            j = int(cat.sample().item())
            neighbors[i, k] = j
            logp_total = logp_total + torch.log(p[j] + 1e-12)
            indeg[j] += 1
            p[j] = 0.0

    logp_total = logp_total / float(N * K)
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
            if s <= 1e-12:
                logp_total = logp_total + torch.log(torch.tensor(1e-12, device=device))
                j = int(neighbors[i, k].item())
                indeg[j] += 1
                p[j] = 0.0
                continue

            p = p / s
            j = int(neighbors[i, k].item())
            logp_total = logp_total + torch.log(p[j] + 1e-12)
            indeg[j] += 1
            p[j] = 0.0

    logp_total = logp_total / float(N * K)
    return logp_total


# =============================================================================
# PPO
# =============================================================================

@dataclass
class PPOConfig:
    horizon: int = 1000
    gamma_fin: float = 5.0
    net_period: int = 5
    K: int = 5
    indeg_cap: int = 8

    n_iters: int = 200
    rollout_len: int = 200
    mini_epochs: int = 5
    minibatch: int = 64

    lr_actor: float = 2e-4
    lr_critic: float = 5e-4

    clip_range: float = 0.2
    target_kl: float = 0.02

    gae_lambda: float = 0.95
    discount: float = 0.99

    entropy_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0

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


def rollout_one(env: InfoNetworkBondEnvV10, actor: ActorNetV10, critic: CriticNet, cfg: PPOConfig):
    device = cfg.device

    obs = env.reset()
    obs_dim = len(obs[0])

    obs_list = []
    r_list = []
    done_list = []

    # logprobs
    logp_net_list = []
    logp_gate_list = []
    logp_total_list = []

    ent_net_list = []
    ent_gate_list = []

    netmask_list = []
    neighbors_list = []
    w_list = []

    gate_u_list = []      # store pre-squash u for PPO evaluation
    gate_g_list = []      # store actual gate g (for env step)

    dbg_last = {}

    for t in range(cfg.rollout_len):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)  # (N,obs_dim)

        # Critic value (not stored here; recomputed later)
        _ = critic(obs_t)

        logits, w_logits, mu, logstd = actor(obs_t)

        # gate action ALWAYS
        g, u, logp_gate, ent_u = gate_sample_and_logprob(mu, logstd)

        # network action only sometimes
        do_net = (env.t % cfg.net_period) == 0
        if do_net:
            neighbors, logp_net = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
            # deterministic weights for stability
            w = torch.softmax(w_logits, dim=-1)

            # diagnostic entropy for network categorical (per-row, mean)
            ent_net = torch.distributions.Categorical(torch.softmax(logits, dim=-1)).entropy().mean()
            netmask = torch.tensor(1.0, device=device)
        else:
            neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)
            logp_net = torch.zeros((), device=device)
            ent_net = torch.zeros((), device=device)
            netmask = torch.tensor(0.0, device=device)

        # total logp used in PPO
        logp_total = logp_gate + netmask * logp_net

        # Step env (numpy)
        if do_net:
            obs, r, done, info = env.step(
                g_gate=g.detach().cpu().numpy(),
                neighbors=neighbors.detach().cpu().numpy(),
                w=w.detach().cpu().numpy(),
            )
        else:
            obs, r, done, info = env.step(g_gate=g.detach().cpu().numpy(), neighbors=None, w=None)

        # store
        obs_list.append(obs_t.detach())
        r_list.append(float(r))
        done_list.append(float(done))

        logp_net_list.append(logp_net.detach())
        logp_gate_list.append(logp_gate.detach())
        logp_total_list.append(logp_total.detach())

        ent_net_list.append(ent_net.detach())
        ent_gate_list.append(ent_u.detach())

        netmask_list.append(netmask.detach())
        neighbors_list.append(neighbors.detach())
        w_list.append(w.detach())

        gate_u_list.append(u.detach())
        gate_g_list.append(g.detach())

        dbg_last = info

        if done:
            break

    batch = {
        "obs": torch.stack(obs_list),  # (T,N,obs_dim)
        "rew": torch.tensor(r_list, dtype=torch.float32, device=device),
        "done": torch.tensor(done_list, dtype=torch.float32, device=device),

        "logp_net": torch.stack(logp_net_list),
        "logp_gate": torch.stack(logp_gate_list),
        "logp_total": torch.stack(logp_total_list),

        "ent_net": torch.stack(ent_net_list),
        "ent_gate": torch.stack(ent_gate_list),

        "netmask": torch.stack(netmask_list),  # (T,)
        "neighbors": neighbors_list,           # list of (N,K)
        "w": w_list,                           # list of (N,K)

        "gate_u": gate_u_list,                 # list of (N,)
        "gate_g": gate_g_list,                 # list of (N,)

        "dbg_last": dbg_last,
        "obs_dim": obs_dim,
    }
    return batch


# =============================================================================
# Baselines + evaluation
# =============================================================================

def run_baseline_fixed(env_ctor, cfg_eval, seeds):
    metrics = []
    for sd in seeds:
        env = env_ctor(sd)
        _ = env.reset()

        returns = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        prices = []
        cum = 0.0
        cum_list = []

        for _ in range(cfg_eval.horizon):
            g = np.ones(env.N, dtype=float)
            _, _, done, info = env.step(g_gate=g, neighbors=None, w=None)

            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])

            cum += info["R"]
            cum_list.append(cum)

            if done:
                break

        rets = np.asarray(returns, dtype=float)
        metrics.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(rets, 0.05)),
            "cumret_min_proxy": float(np.min(np.asarray(cum_list, dtype=float))),
            "final_price": float(prices[-1]) if len(prices) else float(env.p),
        })
    return metrics


def run_baseline_random_rewire(env_ctor, cfg_eval, seeds):
    metrics = []
    for sd in seeds:
        env = env_ctor(sd)
        _ = env.reset()

        returns = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        prices = []
        cum = 0.0
        cum_list = []

        for _ in range(cfg_eval.horizon):
            g = np.ones(env.N, dtype=float)

            if (env.t % cfg_eval.net_period) == 0:
                indeg = np.zeros(env.N, dtype=int)
                neighbors = np.zeros((env.N, cfg_eval.K), dtype=int)
                w = np.zeros((env.N, cfg_eval.K), dtype=float)

                for i in range(env.N):
                    picks = []
                    for _k in range(cfg_eval.K):
                        candidates = [j for j in range(env.N) if j != i and indeg[j] < cfg_eval.indeg_cap and j not in picks]
                        if len(candidates) == 0:
                            candidates = [j for j in range(env.N) if j != i and j not in picks]
                        j = candidates[np.random.randint(0, len(candidates))]
                        picks.append(j)
                        indeg[j] += 1
                    neighbors[i] = np.asarray(picks, dtype=int)
                    w[i] = np.ones(cfg_eval.K, dtype=float) / cfg_eval.K

                _, _, done, info = env.step(g_gate=g, neighbors=neighbors, w=w)
            else:
                _, _, done, info = env.step(g_gate=g, neighbors=None, w=None)

            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])

            cum += info["R"]
            cum_list.append(cum)

            if done:
                break

        rets = np.asarray(returns, dtype=float)
        metrics.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(rets, 0.05)),
            "cumret_min_proxy": float(np.min(np.asarray(cum_list, dtype=float))),
            "final_price": float(prices[-1]) if len(prices) else float(env.p),
        })
    return metrics


def summarize_metrics(name, ms):
    keys = list(ms[0].keys())
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in ms], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std()))
    return name, out


def print_compare_table(rows, title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    keys = list(rows[0][1].keys())
    header = f"{'Metric':<24} | " + " | ".join([f"{nm:<24}" for nm, _ in rows])
    print(header)
    print("-" * 110)
    for k in keys:
        line = f"{k:<24} | " + " | ".join([f"{d[k][0]:>10.6g} ± {d[k][1]:>10.6g}".ljust(24) for _, d in rows])
        print(line)


# =============================================================================
# Train
# =============================================================================

def train_v10():
    cfg = PPOConfig(
        horizon=1000,
        gamma_fin=5.0,
        net_period=5,
        K=5,
        indeg_cap=8,

        n_iters=200,
        rollout_len=200,
        mini_epochs=5,
        minibatch=64,

        lr_actor=2e-4,
        lr_critic=5e-4,

        clip_range=0.2,
        target_kl=0.02,

        gae_lambda=0.95,
        discount=0.99,

        entropy_coef=0.01,
        vf_coef=1.0,
        max_grad_norm=1.0,

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
    )

    print("==============================================================================================================")
    print(f"CONFIG v10: horizon={cfg.horizon} | gamma_fin={cfg.gamma_fin} | K={cfg.K} | indeg_cap={cfg.indeg_cap} | net_period={cfg.net_period}")
    print(f"Reward: w_risk={env_kwargs['w_risk']} w_flow2={env_kwargs['w_flow2']} w_flow4={env_kwargs['w_flow4']} w_gini={env_kwargs['w_gini']} w_net={env_kwargs['w_net']} w_info={env_kwargs['w_info']}")
    print("Gate: stochastic squashed-Gaussian (logprob included in PPO ratio)")
    print("==============================================================================================================")

    seed_train = 0
    set_seed(seed_train)

    env = InfoNetworkBondEnvV10(seed=seed_train, **env_kwargs)
    env.gamma_fin = cfg.gamma_fin

    obs0 = env.reset()
    obs_dim = len(obs0[0])

    actor = ActorNetV10(obs_dim=obs_dim, hidden=128, emb=64, K=cfg.K).to(cfg.device)
    critic = CriticNet(obs_dim=obs_dim, hidden=128).to(cfg.device)

    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    for it in range(cfg.n_iters):
        batch = rollout_one(env, actor, critic, cfg)

        obs_stack = batch["obs"]  # (T,N,obs_dim)
        T = obs_stack.shape[0]

        # Values
        v = torch.zeros((T,), device=cfg.device)
        for t in range(T):
            v[t] = critic(obs_stack[t])

        with torch.no_grad():
            v_next = critic(obs_stack[-1]) * (1.0 - batch["done"][-1])

        adv, ret = compute_gae(
            rew=batch["rew"],
            done=batch["done"],
            v=v.detach(),
            v_next=v_next.detach(),
            discount=cfg.discount,
            lam=cfg.gae_lambda,
        )

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = torch.arange(T, device=cfg.device)

        actor_losses = []
        critic_losses = []
        approx_kls = []
        clipfracs = []
        ratio_means = []
        ratio_stds = []
        ent_means = []
        gn_actor_list = []
        gn_critic_list = []

        logp_old_total_all = batch["logp_total"].detach()

        for _ep in range(cfg.mini_epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                if mb.numel() == 0:
                    continue

                # old
                logp_old_total = logp_old_total_all[mb]
                netmask = batch["netmask"][mb]
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                # recompute new logp_total for same actions
                logp_new_list = []
                ent_list = []

                for _j, t in enumerate(mb.tolist()):
                    logits_t, w_logits_t, mu_t, logstd_t = actor(obs_stack[t])

                    # gate logp for stored u
                    u_old = batch["gate_u"][t].to(cfg.device)
                    logp_gate_new = gate_logprob_from_u(u_old, mu_t, logstd_t)

                    # net logp if net action happened at t
                    if batch["netmask"][t] > 0.5:
                        neigh_t = batch["neighbors"][t].to(cfg.device)
                        logp_net_new = evaluate_logprob_neighbors_masked(
                            logits=logits_t,
                            neighbors=neigh_t,
                            K=cfg.K,
                            indeg_cap=cfg.indeg_cap,
                        )
                        ent_net = torch.distributions.Categorical(torch.softmax(logits_t, dim=-1)).entropy().mean()
                    else:
                        logp_net_new = torch.zeros((), device=cfg.device)
                        ent_net = torch.zeros((), device=cfg.device)

                    # gate entropy (diag Normal) diagnostic
                    ent_gate = (0.5 * (1.0 + LOG2PI + 2.0 * logstd_t)).mean()

                    logp_total_new = logp_gate_new + (batch["netmask"][t] * logp_net_new)

                    logp_new_list.append(logp_total_new)
                    ent_list.append(ent_net + ent_gate)

                logp_new_total = torch.stack(logp_new_list)
                ent_total = torch.stack(ent_list)

                ratio = torch.exp(logp_new_total - logp_old_total)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                ent_bonus = ent_total.mean()
                actor_loss = policy_loss - cfg.entropy_coef * ent_bonus

                # critic
                v_mb = torch.zeros((mb.numel(),), device=cfg.device)
                for j, t in enumerate(mb.tolist()):
                    v_mb[j] = critic(obs_stack[t])
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
                    approx_kl = torch.mean(logp_old_total - logp_new_total)
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_range).float())

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                approx_kls.append(float(approx_kl.item()))
                clipfracs.append(float(clipfrac.item()))
                ratio_means.append(float(ratio.mean().item()))
                ratio_stds.append(float(ratio.std().item()))
                ent_means.append(float(ent_total.mean().item()))

                gn_actor_list.append(grad_norm(actor))
                gn_critic_list.append(grad_norm(critic))

        if it % 10 == 0:
            dbg = batch["dbg_last"]
            print(
                f"iter={it:04d} | "
                f"actor_loss={np.mean(actor_losses):+.4f} | critic_loss={np.mean(critic_losses):.2f} | "
                f"R_mean={float(batch['rew'].mean().item()):+.4f} | R_std={float(batch['rew'].std().item()):.4f} | "
                f"ratio={np.mean(ratio_means):.3f}±{np.mean(ratio_stds):.3f} | "
                f"kl={np.mean(approx_kls):+.4f} | clipfrac={np.mean(clipfracs):.3f} | ent={np.mean(ent_means):.3f} | "
                f"gradA={np.mean(gn_actor_list):.3g} gradC={np.mean(gn_critic_list):.3g} | "
                f"dbg[riskS]={dbg['riskS']:.3g} dbg[flow2]={dbg['flow2']:.3g} dbg[gini]={dbg['gini']:.3g} dbg[dP]={dbg['dP']:.3g} dbg[pred]={dbg['pred_err']:.3g}"
            )
            if np.mean(approx_kls) > 5 * cfg.target_kl:
                print("WARNING: KL too large; reduce lr_actor or clip_range.")

    # -----------------------------
    # Evaluation
    # -----------------------------
    seeds_eval = list(range(30))

    def env_ctor(sd):
        e = InfoNetworkBondEnvV10(seed=sd, **env_kwargs)
        e.gamma_fin = cfg.gamma_fin
        return e

    fixed_ms = run_baseline_fixed(env_ctor, cfg, seeds_eval)
    rand_ms = run_baseline_random_rewire(env_ctor, cfg, seeds_eval)

    # PPO evaluation: stochastic gate + network sampling (same as training)
    ppo_ms = []
    for sd in seeds_eval:
        set_seed(sd)
        envE = env_ctor(sd)
        obs = envE.reset()

        returns = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        prices = []
        cum = 0.0
        cum_list = []

        for _ in range(cfg.horizon):
            obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=cfg.device)
            with torch.no_grad():
                logits, w_logits, mu, logstd = actor(obs_t)
                g, _, _, _ = gate_sample_and_logprob(mu, logstd)

            do_net = (envE.t % cfg.net_period) == 0
            if do_net:
                neighbors, _ = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
                w = torch.softmax(w_logits, dim=-1)
                obs, _, done, info = envE.step(
                    g_gate=g.cpu().numpy(),
                    neighbors=neighbors.cpu().numpy(),
                    w=w.cpu().numpy(),
                )
            else:
                obs, _, done, info = envE.step(g_gate=g.cpu().numpy(), neighbors=None, w=None)

            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])

            cum += info["R"]
            cum_list.append(cum)

            if done:
                break

        rets = np.asarray(returns, dtype=float)
        ppo_ms.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(rets, 0.05)),
            "cumret_min_proxy": float(np.min(np.asarray(cum_list, dtype=float))),
            "final_price": float(prices[-1]) if len(prices) else float(envE.p),
        })

    rows = [
        summarize_metrics("Fixed", fixed_ms),
        summarize_metrics("RandomRewire", rand_ms),
        summarize_metrics("PPO-v10", ppo_ms),
    ]
    print_compare_table(rows, title="EVALUATION v10 (seeds=30)")


if __name__ == "__main__":
    train_v10()
