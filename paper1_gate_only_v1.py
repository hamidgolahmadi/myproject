# -*- coding: utf-8 -*-
"""
Paper 1 — Gate-only PPO on a Fixed Network (N=100, K=10)
=======================================================

What this script gives you (clean + paper-oriented):

STEP 1) Train a Gate-only PPO policy on a fixed random Top-K network.
STEP 2) Evaluate properly as a *risk–return tradeoff* (not just "minimize tail loss"):
        - Mean PnL return (mu)
        - Vol / Sharpe proxy
        - VaR / CVaR on PnL returns
        - Turnover
STEP 3) Compute constant-gate baselines and build a frontier table (points).
STEP 4) Save everything (config, training history, model, frontier.json).

Important design choices:
- The environment keeps your stabilizing shaping reward (risk/flow/gini),
  but evaluation is economic: PnL returns / Sharpe / tail risk.
- We define PnL return as portfolio return using previous holdings:
    pnl_ret_t = (x_{t-1} * (p_t - p_{t-1}) - tc_t) / (|x_{t-1}|*p_{t-1} + eps)
  This avoids "wealth CVaR" nonsense and reduces trivial accounting bugs.
- Constant gate baselines sweep g in [0,1] and report the best points.

Run:
  python paper1_gate_only_v2.py

"""

import math
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Utils
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

def clip_value(x: float, cap: float | None):
    if cap is None or cap <= 0:
        return float(x)
    return float(np.clip(x, -cap, cap))

def tail_stats(x: np.ndarray, alpha: float = 0.05):
    """
    Returns: VaR (quantile), CVaR (mean of tail), min, tail_n
    Tail is the lower tail (bad outcomes).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(var=np.nan, cvar=np.nan, min=np.nan, tail_n=0)
    q = float(np.quantile(x, alpha))
    tail = x[x <= q]
    c = float(tail.mean()) if tail.size else q
    m = float(x.min())
    return dict(var=q, cvar=c, min=m, tail_n=int(tail.size))

def sharpe_proxy(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x) / (np.std(x) + eps))


# =============================================================================
# Environment (Fixed Network)
# =============================================================================

class FixedNetworkMarketEnv:
    """
    Fixed network environment for Paper 1 (Gate-only).
    - P fixed for the entire episode
    - Gate g_t ∈ [0,1] scales responsiveness to social signal

    Training reward = shaping for stability (not economic):
        r_t = - w_risk*riskS - w_flow2*flow2 - w_flow4*flow4 - w_gini*gini_in

    Economic evaluation metrics logged separately:
        - pnl_ret_t (portfolio return proxy)
        - turnover
        - price returns, etc.
    """

    def __init__(
        self,
        # size
        N=100,
        K=10,
        horizon=1000,
        # price & trading
        p0=100.0,
        kappa=0.02,
        sigma_eps=0.1,
        x_max=1.0,
        tau=0.001,
        # latent / signals / belief
        rho_y=0.98,
        sigma_y=0.02,
        sigma_s=0.05,
        omega_social=0.7,
        sigma_belief=0.02,
        # risk EWMA
        beta_risk=None,
        # shaping weights
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=0.5,
        cap_riskS=50.0,
        cap_flow4=5.0,
        cap_reward=50.0,
        seed=0,
    ):
        self.N = int(N)
        self.K = int(K)
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

        self.cap_riskS = float(cap_riskS) if cap_riskS is not None else None
        self.cap_flow4 = float(cap_flow4) if cap_flow4 is not None else None
        self.cap_reward = float(cap_reward) if cap_reward is not None else None

        self.rng = np.random.default_rng(seed)

        # responsiveness scale inside tanh (paper parameter)
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
        return self.y + self.rng.normal(0.0, self.sigma_s, size=self.N)

    def _build_obs(self):
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)

        Pb = self.P @ self.b
        var_b = float(np.var(self.b))
        vol = float(np.sqrt(max(self.risk_v, 0.0)))
        mean_abs_deltax = float(np.mean(np.abs(self.delta_x_prev)))

        # per-agent observation
        obs = []
        for i in range(self.N):
            obs.append(np.array([
                self.p,            # price level
                self.x[i],         # position
                self.cash[i],      # cash
                self.b[i],         # belief
                self.s[i],         # private signal
                Pb[i],             # social belief agg
                in_deg[i],         # indegree
                var_b,             # dispersion
                self.R_prev,       # last price return
                vol,               # risk proxy
                mean_abs_deltax,   # turnover proxy
                gini_in,           # indegree inequality
            ], dtype=float))
        return obs

    def reset(self):
        self.t = 0
        self.p = self.p0
        self.p_prev = self.p0
        self.R_prev = 0.0

        # positions and cash
        self.x = self.rng.normal(0.0, 0.1, size=self.N)
        self.cash = np.zeros(self.N, dtype=float)

        # fixed network for the episode
        self.P = self._init_P_random_topk()

        # latent + signals + beliefs
        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.s = self._private_signals()
        self.b = self.rng.normal(0.0, 0.2, size=self.N)

        # risk proxy
        self.risk_v = 0.0
        self.delta_x_prev = np.zeros(self.N, dtype=float)

        # economic series
        self.pnl_ret_series = []
        self.price_ret_series = []
        self.turnover_series = []
        self.tc_series = []

        return self._build_obs()

    def step(self, gate: np.ndarray):
        gate = np.clip(np.asarray(gate, dtype=float), 0.0, 1.0)

        # latent evolves
        self.y = self.rho_y * self.y + float(self.rng.normal(0.0, self.sigma_y))
        self.s = self._private_signals()

        # belief update (DeGroot-ish)
        Pb_old = self.P @ self.b
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + \
                 self.rng.normal(0.0, self.sigma_belief, size=self.N)

        # action: responsiveness to social signal
        social_signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * gate * social_signal)      # [-1,1]
        delta_x = a_fin * self.x_max                                # trade size

        # transaction cost
        tc = self.tau * np.abs(delta_x)

        # economic pnl return computed on *previous holdings*
        x_prev = self.x.copy()
        p_prev = float(self.p)

        # update positions
        self.x = self.x + delta_x

        # price formation with impact
        net_flow = float(np.sum(delta_x))
        self.p_prev = self.p
        self.p = self.p + self.kappa * net_flow + float(self.rng.normal(0.0, self.sigma_eps))
        self.p = max(self.p, 1e-6)

        # price return
        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        # risk EWMA
        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        # cash update (execution at new price is a simplification; ok for toy model)
        self.cash = self.cash - self.p * delta_x - tc

        # pnl return (portfolio return proxy)
        # PnL from price move on previous holdings minus costs, normalized by exposure
        exposure = np.abs(x_prev) * p_prev
        denom = float(np.mean(exposure) + 1e-12)
        pnl = float(np.mean(x_prev) * (self.p - p_prev) - np.mean(tc))
        pnl_ret = pnl / denom

        # trackers
        self.delta_x_prev = delta_x.copy()
        self.pnl_ret_series.append(pnl_ret)
        self.price_ret_series.append(float(R))
        self.turnover_series.append(float(np.mean(np.abs(delta_x))))
        self.tc_series.append(float(np.mean(tc)))

        # shaping reward
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)

        flow2 = float(np.mean(delta_x ** 2))
        flow4 = float(np.mean(delta_x ** 4))

        riskS = float(self.risk_v / max(self.risk_unit, 1e-18))
        riskS_c = clip_value(riskS, self.cap_riskS)
        flow4_c = float(np.clip(flow4, 0.0, self.cap_flow4)) if self.cap_flow4 else flow4

        r = 0.0
        r -= self.w_risk * riskS_c
        r -= self.w_flow2 * flow2
        r -= self.w_flow4 * flow4_c
        r -= self.w_gini * gini_in

        if self.cap_reward is not None and self.cap_reward > 0:
            r = float(np.clip(r, -self.cap_reward, self.cap_reward))

        self.t += 1
        done = self.t >= self.horizon

        info = {
            "price": float(self.p),
            "R_price": float(R),
            "pnl_ret": float(pnl_ret),
            "turnover": float(np.mean(np.abs(delta_x))),
            "tc": float(np.mean(tc)),
            "riskS": float(riskS),
            "flow2": float(flow2),
            "flow4": float(flow4),
            "gini": float(gini_in),
            "gate_mean": float(np.mean(gate)),
        }
        return self._build_obs(), float(r), done, info


# =============================================================================
# Policy: Squashed Gaussian Gate
# =============================================================================

LOG_2PI = float(math.log(2.0 * math.pi))

def gaussian_logprob(z: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
    var = torch.exp(2.0 * logstd)
    return -0.5 * ((z - mu) ** 2 / (var + 1e-8) + 2.0 * logstd + LOG_2PI)

def sample_squashed_gaussian_gate(mu: torch.Tensor, logstd: torch.Tensor):
    """
    Sample z ~ N(mu, std), squash with tanh to u in (-1,1), then map to gate g in (0,1).
    Returns:
      g: (N,)
      logp_norm: scalar (mean log-prob across agents)
      ent_mean: scalar
    """
    std = torch.exp(logstd)
    z = mu + std * torch.randn_like(mu)
    u = torch.tanh(z)
    g = 0.5 * (u + 1.0)

    logp_z = gaussian_logprob(z, mu, logstd)
    # change-of-variables correction for tanh
    log_det = torch.log(1.0 - u * u + 1e-6)
    logp_u = logp_z - log_det
    # correction for scaling u -> g = 0.5(u+1)
    logp_g = logp_u + math.log(2.0)
    logp_norm = logp_g.mean()

    ent = 0.5 * (LOG_2PI + 1.0) + logstd
    ent_mean = ent.mean()
    return g, logp_norm, ent_mean

def logprob_squashed_gaussian_gate(g: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
    """
    Log-prob of g in (0,1) for the squashed Gaussian construction.
    Returns scalar = mean log-prob across agents.
    """
    g = torch.clamp(g, 1e-6, 1.0 - 1e-6)
    u = 2.0 * g - 1.0
    u = torch.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6)

    # atanh(u)
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
        mu = self.to_mu(h).squeeze(-1)            # (N,)
        raw = self.to_logstd_raw(h).squeeze(-1)   # (N,)
        s = torch.sigmoid(raw)
        logstd = logstd_min + (logstd_max - logstd_min) * s
        return mu, logstd

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        pooled = obs_t.mean(dim=0, keepdim=True)  # (1, obs_dim)
        return self.v(pooled).squeeze(-1)         # scalar


# =============================================================================
# PPO Helpers
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
    v_next_t = v_next
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = rew[t] + discount * v_next_t * mask - v[t]
        gae = delta + discount * lam * mask * gae
        adv[t] = gae
        v_next_t = v[t]
    ret = adv + v
    return adv, ret

def grad_norm(module: nn.Module):
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().data.norm(2).item()
        total += g * g
    return float(math.sqrt(total))


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # env
    N: int = 100
    K: int = 10
    horizon: int = 1000
    gamma_fin: float = 5.0

    # rollout/train
    rollout_len: int = 200
    n_iters: int = 300

    # PPO
    lr_actor: float = 5e-5
    lr_critic: float = 5e-4
    clip: float = 0.1
    mini_epochs: int = 2
    minibatch: int = 64
    discount: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.005

    # mild interior bias (optional; keep small)
    w_gate_mean: float = 0.2
    gate_target: float = 0.4

    # gate distribution bounds
    logstd_min: float = -2.5
    logstd_max: float = -0.5

    # critic stability
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    ret_clip: float = 20.0
    huber_delta: float = 10.0

    # device/logging
    device: str = "cpu"
    seed: int = 0

    # evaluation
    eval_episodes: int = 5
    tail_alpha: float = 0.05

    # constant-gate baseline sweep
    const_gates: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


# =============================================================================
# Rollout
# =============================================================================

def rollout_one(env: FixedNetworkMarketEnv, actor: GateActor, critic: Critic, cfg: Config):
    device = cfg.device
    obs = env.reset()

    obs_list, done_list, rew_list = [], [], []
    gate_list, logp_list, ent_list = [], [], []
    v_list = []

    # econ trackers
    pnl_ret_list = []
    price_ret_list = []
    turnover_list = []
    gate_mean_list = []

    for _ in range(cfg.rollout_len):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)  # (N, obs_dim)
        v = critic(obs_t)

        mu, logstd = actor(obs_t, cfg.logstd_min, cfg.logstd_max)
        gate, logp, ent = sample_squashed_gaussian_gate(mu, logstd)

        obs, r, done, info = env.step(gate.detach().cpu().numpy())

        obs_list.append(obs_t)
        v_list.append(v.squeeze())
        done_list.append(float(done))
        rew_list.append(float(r))

        gate_list.append(gate)
        logp_list.append(logp)
        ent_list.append(ent)

        pnl_ret_list.append(float(info["pnl_ret"]))
        price_ret_list.append(float(info["R_price"]))
        turnover_list.append(float(info["turnover"]))
        gate_mean_list.append(float(info["gate_mean"]))

        if done:
            break

    batch = {
        "obs": torch.stack(obs_list),  # (T, N, obs_dim)
        "done": torch.tensor(done_list, dtype=torch.float32, device=device),  # (T,)
        "rew": torch.tensor(rew_list, dtype=torch.float32, device=device),    # (T,)
        "v": torch.stack(v_list),                                            # (T,)
        "gate": torch.stack(gate_list),                                      # (T, N)
        "logp": torch.stack(logp_list),                                      # (T,)
        "ent": torch.stack(ent_list),                                        # (T,)
        "econ": {
            "pnl_ret": np.asarray(pnl_ret_list, dtype=float),
            "price_ret": np.asarray(price_ret_list, dtype=float),
            "turnover": np.asarray(turnover_list, dtype=float),
            "gate_mean": np.asarray(gate_mean_list, dtype=float),
        }
    }
    return batch


# =============================================================================
# Evaluation (economic metrics + baselines + frontier)
# =============================================================================

@torch.no_grad()
def run_episode_policy(env: FixedNetworkMarketEnv, actor: GateActor, cfg: Config, deterministic: bool = True):
    obs = env.reset()
    pnl_ret = []
    price_ret = []
    turnover = []
    gate_mean = []

    for _ in range(env.horizon):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=cfg.device)
        mu, logstd = actor(obs_t, cfg.logstd_min, cfg.logstd_max)
        if deterministic:
            u = torch.tanh(mu)
            g = 0.5 * (u + 1.0)
        else:
            g, _, _ = sample_squashed_gaussian_gate(mu, logstd)

        obs, _, done, info = env.step(g.detach().cpu().numpy())
        pnl_ret.append(float(info["pnl_ret"]))
        price_ret.append(float(info["R_price"]))
        turnover.append(float(info["turnover"]))
        gate_mean.append(float(info["gate_mean"]))
        if done:
            break

    pnl_ret = np.asarray(pnl_ret, dtype=float)
    price_ret = np.asarray(price_ret, dtype=float)
    return pnl_ret, price_ret, np.asarray(turnover, dtype=float), np.asarray(gate_mean, dtype=float)

def eval_actor(env_kwargs: dict, cfg: Config, actor: GateActor):
    actor.eval()
    out = []
    for ep in range(cfg.eval_episodes):
        env = FixedNetworkMarketEnv(**env_kwargs, seed=cfg.seed + 10_000 + ep)
        env.gamma_fin = cfg.gamma_fin
        pnl_ret, price_ret, turnover, gmean = run_episode_policy(env, actor, cfg, deterministic=True)

        ts = tail_stats(pnl_ret, alpha=cfg.tail_alpha)
        out.append({
            "mu_pnl": float(np.mean(pnl_ret)),
            "sig_pnl": float(np.std(pnl_ret)),
            "sharpe_pnl": sharpe_proxy(pnl_ret),
            "VaR_pnl": float(ts["var"]),
            "CVaR_pnl": float(ts["cvar"]),
            "CVaR_Loss": float(-ts["cvar"]),   # Loss = -CVaR
            "Min_pnl": float(ts["min"]),
            "TailN": int(ts["tail_n"]),
            "mu_priceR": float(np.mean(price_ret)),
            "sig_priceR": float(np.std(price_ret)),
            "turnover": float(np.mean(turnover)),
            "gate_mean": float(np.mean(gmean)),
        })

    # aggregate
    keys = out[0].keys()
    agg = {k: float(np.mean([d[k] for d in out])) for k in keys}
    agg_std = {k + "_stdAcrossEps": float(np.std([d[k] for d in out])) for k in keys}
    agg.update(agg_std)
    actor.train()
    return agg

def eval_constant_gate(env_kwargs: dict, cfg: Config, gate_value: float):
    """
    Evaluate fixed constant gate g for the whole episode.
    """
    out = []
    for ep in range(cfg.eval_episodes):
        env = FixedNetworkMarketEnv(**env_kwargs, seed=cfg.seed + 20_000 + ep)
        env.gamma_fin = cfg.gamma_fin
        obs = env.reset()
        pnl_ret = []
        price_ret = []
        turnover = []
        gmean = []

        g = np.full((env.N,), float(gate_value), dtype=float)
        for _ in range(env.horizon):
            obs, _, done, info = env.step(g)
            pnl_ret.append(float(info["pnl_ret"]))
            price_ret.append(float(info["R_price"]))
            turnover.append(float(info["turnover"]))
            gmean.append(float(info["gate_mean"]))
            if done:
                break

        pnl_ret = np.asarray(pnl_ret, dtype=float)
        ts = tail_stats(pnl_ret, alpha=cfg.tail_alpha)
        out.append({
            "gate_const": float(gate_value),
            "mu_pnl": float(np.mean(pnl_ret)),
            "sig_pnl": float(np.std(pnl_ret)),
            "sharpe_pnl": sharpe_proxy(pnl_ret),
            "CVaR_Loss": float(-ts["cvar"]),
            "CVaR_pnl": float(ts["cvar"]),
            "VaR_pnl": float(ts["var"]),
            "Min_pnl": float(ts["min"]),
            "TailN": int(ts["tail_n"]),
            "turnover": float(np.mean(turnover)),
        })

    # aggregate
    keys = out[0].keys()
    agg = {k: float(np.mean([d[k] for d in out])) for k in keys}
    agg_std = {k + "_stdAcrossEps": float(np.std([d[k] for d in out])) for k in keys}
    agg.update(agg_std)
    return agg

def build_frontier(env_kwargs: dict, cfg: Config, actor: GateActor):
    """
    Returns a dict with:
      - ppo_eval: aggregated eval of PPO actor
      - const_points: list of aggregated points for constant gates
      - best_by_loss: constant gate with minimum CVaR_Loss
      - best_by_sharpe: constant gate with maximum sharpe_pnl
    """
    ppo_eval = eval_actor(env_kwargs, cfg, actor)

    const_points = []
    for g in cfg.const_gates:
        const_points.append(eval_constant_gate(env_kwargs, cfg, float(g)))

    best_by_loss = min(const_points, key=lambda d: d["CVaR_Loss"])
    best_by_sharpe = max(const_points, key=lambda d: d["sharpe_pnl"])

    return {
        "ppo_eval": ppo_eval,
        "const_points": const_points,
        "best_const_by_loss": best_by_loss,
        "best_const_by_sharpe": best_by_sharpe,
    }


# =============================================================================
# Train
# =============================================================================

def train(cfg: Config, env_kwargs: dict, out_dir: str = "./paper1_runs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_path = Path(out_dir) / f"run_{run_id}"
    run_path.mkdir(parents=True, exist_ok=True)

    # save config
    with open(run_path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"cfg": asdict(cfg), "env_kwargs": env_kwargs}, f, indent=2)

    set_seed(cfg.seed)

    # build env just for obs_dim
    env = FixedNetworkMarketEnv(**env_kwargs, seed=cfg.seed)
    env.gamma_fin = cfg.gamma_fin
    obs0 = env.reset()
    obs_dim = len(obs0[0])

    actor = GateActor(obs_dim=obs_dim, hidden=128).to(cfg.device)
    critic = Critic(obs_dim=obs_dim, hidden=128).to(cfg.device)

    optA = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    optV = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    history = []
    best_eval_loss = float("inf")
    best_model_state = None

    print("=" * 110)
    print("Paper1 Gate-only PPO | Fixed Network | STEP 2: Economic eval + Constant-gate frontier")
    print(f"N={cfg.N}, K={cfg.K}, horizon={cfg.horizon}, gamma_fin={cfg.gamma_fin}")
    print(f"PPO: lrA={cfg.lr_actor}, lrV={cfg.lr_critic}, clip={cfg.clip}, ent={cfg.ent_coef}")
    print("=" * 110)

    for it in range(cfg.n_iters):
        batch = rollout_one(env, actor, critic, cfg)

        obs = batch["obs"]
        done = batch["done"]
        rew  = batch["rew"]
        v    = batch["v"].detach()
        logp_old = batch["logp"].detach()
        gate_old = batch["gate"].detach()

        T = obs.shape[0]
        with torch.no_grad():
            v_next = critic(obs[-1]) * (1.0 - done[-1])

        adv, ret = compute_gae(rew=rew, done=done, v=v, v_next=v_next,
                               discount=cfg.discount, lam=cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        ret = torch.clamp(ret, -cfg.ret_clip, cfg.ret_clip)

        idx = torch.arange(T, device=cfg.device)

        lossA_list, lossV_list = [], []
        ratio_m_list, ratio_s_list, kl_list, clipfrac_list = [], [], [], []
        ent_list = []
        gnA_list, gnV_list = [], []

        for _ in range(cfg.mini_epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                if mb.numel() == 0:
                    continue

                logp_new = []
                ent_mb = []
                v_mb = torch.zeros((mb.numel(),), device=cfg.device)

                # recompute in a loop (keep it explicit and correct; optimize later)
                for j, t in enumerate(mb.tolist()):
                    obs_t = obs[t]
                    v_mb[j] = critic(obs_t)

                    mu, logstd = actor(obs_t, cfg.logstd_min, cfg.logstd_max)
                    lp = logprob_squashed_gaussian_gate(gate_old[t], mu, logstd)
                    logp_new.append(lp)

                    ent = (0.5 * (LOG_2PI + 1.0) + logstd).mean()
                    ent_mb.append(ent)

                logp_new = torch.stack(logp_new)       # (mb,)
                ent_mb = torch.stack(ent_mb)           # (mb,)

                ratio = torch.exp(logp_new - logp_old[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv[mb]
                pg_loss = -torch.mean(torch.min(surr1, surr2))

                # mild interior bias (optional; keep small)
                g_mean = gate_old[mb].mean()
                gate_mean_pen = cfg.w_gate_mean * (g_mean - cfg.gate_target) ** 2

                ent_bonus = torch.mean(ent_mb)
                lossA = pg_loss - cfg.ent_coef * ent_bonus + gate_mean_pen

                td = v_mb - ret[mb]
                lossV = torch.mean(huber_loss(td, cfg.huber_delta))

                optA.zero_grad(set_to_none=True)
                lossA.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                optA.step()

                optV.zero_grad(set_to_none=True)
                (cfg.vf_coef * lossV).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                optV.step()

                with torch.no_grad():
                    approx_kl = torch.mean(logp_old[mb] - logp_new)
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip).float())

                lossA_list.append(float(lossA.item()))
                lossV_list.append(float(lossV.item()))
                ratio_m_list.append(float(ratio.mean().item()))
                ratio_s_list.append(float(ratio.std(unbiased=False).item()))
                kl_list.append(float(approx_kl.item()))
                clipfrac_list.append(float(clipfrac.item()))
                ent_list.append(float(ent_bonus.item()))
                gnA_list.append(grad_norm(actor))
                gnV_list.append(grad_norm(critic))

        # log + eval every 10 iters
        if it % 10 == 0:
            econ = batch["econ"]
            pnl_seq = econ["pnl_ret"]
            price_seq = econ["price_ret"]

            ts_pnl = tail_stats(pnl_seq, alpha=cfg.tail_alpha)
            ts_price = tail_stats(price_seq, alpha=cfg.tail_alpha)

            eval_pack = build_frontier(env_kwargs, cfg, actor)
            ppo_eval = eval_pack["ppo_eval"]
            best_const = eval_pack["best_const_by_loss"]

            row = {
                "iter": it,
                "train_rew_mean": float(batch["rew"].mean().item()),
                "train_rew_std": float(batch["rew"].std(unbiased=False).item()),
                "train_gate_mean": float(np.mean(econ["gate_mean"])),
                "train_turnover": float(np.mean(econ["turnover"])),

                "train_mu_pnl": float(np.mean(pnl_seq)),
                "train_sig_pnl": float(np.std(pnl_seq)),
                "train_sharpe_pnl": sharpe_proxy(pnl_seq),
                "train_VaR_pnl": float(ts_pnl["var"]),
                "train_CVaR_pnl": float(ts_pnl["cvar"]),
                "train_CVaR_Loss": float(-ts_pnl["cvar"]),
                "train_Min_pnl": float(ts_pnl["min"]),
                "train_TailN_pnl": int(ts_pnl["tail_n"]),

                "train_VaR_price": float(ts_price["var"]),
                "train_CVaR_price": float(ts_price["cvar"]),
                "train_Min_price": float(ts_price["min"]),
                "train_TailN_price": int(ts_price["tail_n"]),

                "lossA": float(np.mean(lossA_list)) if lossA_list else 0.0,
                "lossV": float(np.mean(lossV_list)) if lossV_list else 0.0,
                "ratio": float(np.mean(ratio_m_list)) if ratio_m_list else 1.0,
                "ratio_std": float(np.mean(ratio_s_list)) if ratio_s_list else 0.0,
                "kl": float(np.mean(kl_list)) if kl_list else 0.0,
                "clipfrac": float(np.mean(clipfrac_list)) if clipfrac_list else 0.0,
                "ent": float(np.mean(ent_list)) if ent_list else 0.0,
                "gradA": float(np.mean(gnA_list)) if gnA_list else 0.0,
                "gradV": float(np.mean(gnV_list)) if gnV_list else 0.0,

                # PPO eval summary
                "eval_mu_pnl": float(ppo_eval["mu_pnl"]),
                "eval_sharpe_pnl": float(ppo_eval["sharpe_pnl"]),
                "eval_CVaR_Loss": float(ppo_eval["CVaR_Loss"]),
                "eval_gate_mean": float(ppo_eval["gate_mean"]),
                "eval_turnover": float(ppo_eval["turnover"]),

                # best constant-gate by loss
                "best_const_gate": float(best_const["gate_const"]),
                "best_const_loss": float(best_const["CVaR_Loss"]),
                "best_const_sharpe": float(best_const["sharpe_pnl"]),
            }
            history.append(row)

            print(
                f"iter={it:04d} | "
                f"rew={row['train_rew_mean']:+.3f}±{row['train_rew_std']:.3f} | "
                f"gate={row['train_gate_mean']:.3f} turn={row['train_turnover']:.3f} | "
                f"PnL Tail: VaR={row['train_VaR_pnl']:+.5f} CVaR={row['train_CVaR_pnl']:+.5f} "
                f"Min={row['train_Min_pnl']:+.5f} N={row['train_TailN_pnl']} | "
                f"PPO: lossA={row['lossA']:+.3f} kl={row['kl']:+.4f} ratio={row['ratio']:.3f}±{row['ratio_std']:.3f} "
                f"clip={row['clipfrac']:.3f} ent={row['ent']:+.3f} | "
                f"EVAL: Sharpe={row['eval_sharpe_pnl']:+.3f} Loss={row['eval_CVaR_Loss']:+.5f} gate={row['eval_gate_mean']:.3f} | "
                f"BEST CONST g={row['best_const_gate']:.1f} Loss={row['best_const_loss']:+.5f} Sharpe={row['best_const_sharpe']:+.3f}"
            )

            # track best by eval tail loss (you can change selection criterion later)
            if row["eval_CVaR_Loss"] < best_eval_loss:
                best_eval_loss = row["eval_CVaR_Loss"]
                best_model_state = {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "iter": it,
                    "eval_CVaR_Loss": best_eval_loss,
                }

    # save final model + best model + logs + frontier
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, run_path / "model_final.pt")
    if best_model_state is not None:
        torch.save(best_model_state, run_path / "model_best_by_loss.pt")

    with open(run_path / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # final frontier package
    frontier = build_frontier(env_kwargs, cfg, actor)
    with open(run_path / "frontier.json", "w", encoding="utf-8") as f:
        json.dump(frontier, f, indent=2)

    return str(run_path)


# =============================================================================
# Entry
# =============================================================================

if __name__ == "__main__":
    cfg = Config()

    env_kwargs = dict(
        N=cfg.N,
        K=cfg.K,
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
        beta_risk=None,
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=0.5,
        cap_riskS=50.0,
        cap_flow4=5.0,
        cap_reward=50.0,
    )

    run_path = train(cfg, env_kwargs, out_dir="./paper1_runs")
    print(f"Saved run to: {run_path}")
