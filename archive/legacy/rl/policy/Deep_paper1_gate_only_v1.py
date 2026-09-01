# -*- coding: utf-8 -*-
"""
Paper1 - Gate-only PPO | Fixed Network Market (Economic Reward) - FIXED VERSION
------------------------------------------------------------------------------
Fixes applied (real fixes):
1) dataclass mutable default fixed_gates -> default_factory
2) Correct PPO bootstrap/GAE using V(s_{t+1}) from next_obs for ALL t (v_next)
3) Removed in-env per-step std normalization (non-stationary scaling).
   Added RunningMeanStd normalizer outside env and applied consistently.
"""

import time
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta


# ===================== CONFIG =====================
@dataclass
class Config:
    # Core
    N: int = 100
    K: int = 10
    T: int = 1000
    episodes: int = 300
    eval_runs: int = 10

    # Fundamental AR(1)
    rho_y: float = 0.98
    sigma_y: float = 0.05
    y0: float = 0.0

    # Signals & beliefs
    sigma_s: float = 0.10
    omega: float = 0.70
    sigma_b: float = 0.05

    # Action -> position
    x_max: float = 1.0
    gamma_fin: float = 2.0

    # Transaction cost
    tau: float = 0.0005

    # Price
    alpha: float = 0.01
    kappa: float = 0.001
    sigma_eps: float = 0.005
    p0: float = 100.0

    # Return normalization
    E_min: float = 1.0

    # Risk EWMA
    beta_ewma: float = 0.95

    # Economic reward weights
    beta_risk: float = 0.5
    lambda_turn: float = 0.05
    w_return: float = 0.1

    # PPO
    lrA: float = 3e-4
    lrV: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    ppo_epochs: int = 8
    batchA: int = 1024
    batchV: int = 128

    # Networks
    actor_hidden: Tuple[int, int] = (128, 64)
    critic_hidden: Tuple[int, int, int] = (128, 64, 32)

    # Eval
    cvar_alpha: float = 0.05
    fixed_gates: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.0, 11))

    # Random / device
    seed: int = 42
    device: str = "cpu"

    # Economic checks
    min_acceptable_return: float = 0.0
    trivial_gate_threshold: float = 0.1


CFG = Config()


# ===================== UTILS =====================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_row_stochastic_topk(N: int, K: int, rng: np.random.Generator):
    P = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        cand = np.array([j for j in range(N) if j != i], dtype=int)
        nbrs = rng.choice(cand, size=K, replace=False)
        P[i, nbrs] = 1.0 / K
    in_deg = (P > 0).sum(axis=0).astype(np.float64)
    return P, in_deg


def tail_metrics(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"VaR": np.nan, "CVaR": np.nan, "Min": np.nan, "TailN": 0}
    q = float(np.quantile(x, alpha))
    tail = x[x <= q]
    cvar = float(tail.mean()) if tail.size > 0 else q
    return {"VaR": q, "CVaR": cvar, "Min": float(x.min()), "TailN": int(tail.size)}


def sharpe_annualized(r: np.ndarray, scale: float = 252.0):
    r = np.asarray(r, dtype=np.float64)
    if r.size < 2:
        return 0.0
    s = r.std()
    if s <= 1e-12:
        return 0.0
    return float(r.mean() / s * np.sqrt(scale))


def calculate_cumulative_return(returns: np.ndarray):
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


# ===================== NORMALIZER =====================
class RunningMeanStd:
    """
    Running mean/std for vector observations.
    Update with batches; normalize consistently.
    """
    def __init__(self, shape, eps=1e-4, clip=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.clip = clip

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = tot_count

    def normalize(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var).astype(np.float32)
        y = (x - self.mean.astype(np.float32)) / (std + 1e-8)
        return np.clip(y, -self.clip, self.clip).astype(np.float32)


# ===================== ENV =====================
class NetworkMarketEnv:
    """
    Fixed network market environment with proper economic accounting.
    Obs is NOT self-normalized by per-step std (removed).
    """
    def __init__(self, cfg: Config, resample_each_episode: bool = False):
        self.cfg = cfg
        self.N = cfg.N
        self.resample_each_episode = resample_each_episode
        self.rng = np.random.default_rng(cfg.seed)

        self.P = None
        self.in_deg = None
        self._maybe_build_network()
        self.reset()

    def _maybe_build_network(self):
        if (self.P is None) or self.resample_each_episode:
            self.P, self.in_deg = generate_row_stochastic_topk(self.cfg.N, self.cfg.K, self.rng)

    def reset(self):
        self._maybe_build_network()

        self.t = 0
        self.y = float(self.cfg.y0)
        self.p = float(self.cfg.p0)
        self.p_log_return = 0.0

        self.x = np.zeros(self.N, dtype=np.float64)
        self.b = self.rng.normal(0.0, 0.1, self.N).astype(np.float64)

        self.s = np.zeros(self.N, dtype=np.float64)
        self.m = (self.P @ self.b).astype(np.float64)

        self.var_ewma = 1e-4
        self.last_price_ret = 0.0
        self.last_R_port = 0.0
        self.delta_x_prev = np.zeros(self.N, dtype=np.float64)

        self.R_port_hist = []
        self.price_ret_hist = []
        self.turn_hist = []
        self.gate_mean_hist = []
        self.wealth = 100.0
        self.wealth_hist = [self.wealth]

        self._update_fundamental_and_signals()
        self._update_beliefs()
        return self._get_obs()

    def _update_fundamental_and_signals(self):
        eps_y = self.rng.normal(0.0, self.cfg.sigma_y)
        self.y = float(self.cfg.rho_y * self.y + eps_y)
        eta = self.rng.normal(0.0, self.cfg.sigma_s, self.N)
        self.s = (self.y + eta).astype(np.float64)

    def _update_beliefs(self):
        noise_b = self.rng.normal(0.0, self.cfg.sigma_b, self.N)
        self.b = ((1.0 - self.cfg.omega) * self.s +
                  self.cfg.omega * (self.P @ self.b) +
                  noise_b).astype(np.float64)
        self.m = (self.P @ self.b).astype(np.float64)

    def step(self, g_actions: np.ndarray):
        g = np.asarray(g_actions, dtype=np.float64).reshape(-1)
        if g.size != self.N:
            raise ValueError(f"Gate actions must be shape ({self.N},), got {g.shape}")
        g = np.clip(g, 0.0, 1.0)

        p_prev = float(self.p)
        x_prev = self.x.copy()

        scaled = self.cfg.gamma_fin * g * self.m
        delta_x = self.cfg.x_max * np.tanh(scaled)
        tc = self.cfg.tau * np.abs(delta_x)

        impact = self.cfg.kappa * float(delta_x.sum())
        eps_p = float(self.rng.normal(0.0, self.cfg.sigma_eps))
        p_new = p_prev * (1.0 + self.cfg.alpha * self.y) + impact + eps_p
        p_new = max(p_new, 0.1)
        price_change = p_new - p_prev

        self.p = float(p_new)
        self.p_log_return = float(np.log(self.p / (p_prev + 1e-12)))
        self.x = (x_prev + delta_x).astype(np.float64)

        # next state dynamics
        self._update_fundamental_and_signals()
        self._update_beliefs()

        R_price = float(price_change / (p_prev + 1e-12))
        self.last_price_ret = R_price

        # No look-ahead PnL
        pnl_i = x_prev * price_change - tc

        denom_i = np.maximum(np.abs(x_prev) * p_prev, self.cfg.E_min)
        denom = float(denom_i.sum())
        R_port = float(pnl_i.sum() / (denom + 1e-12))
        self.last_R_port = R_port

        self.wealth *= (1.0 + R_port)
        self.wealth_hist.append(self.wealth)

        turn = float(np.mean(np.abs(delta_x)))

        self.var_ewma = float(self.cfg.beta_ewma * self.var_ewma +
                              (1.0 - self.cfg.beta_ewma) * (R_price ** 2))

        positive_return = max(0.0, R_port)
        reward = (R_port +
                  self.cfg.w_return * positive_return -
                  self.cfg.beta_risk * self.var_ewma -
                  self.cfg.lambda_turn * turn)

        self.delta_x_prev = delta_x.copy()
        self.R_port_hist.append(R_port)
        self.price_ret_hist.append(R_price)
        self.turn_hist.append(turn)
        self.gate_mean_hist.append(float(g.mean()))

        self.t += 1
        done = (self.t >= self.cfg.T)

        info = {
            "R_port": R_port,
            "R_price": R_price,
            "var_ewma": self.var_ewma,
            "turnover": turn,
            "mean_gate": float(g.mean()),
            "price": self.p,
            "p_prev": p_prev,
            "wealth": self.wealth,
            "cumulative_return": (self.wealth - 100.0) / 100.0,
        }
        return self._get_obs(), float(reward), done, info

    def _get_obs(self):
        indeg_scaled = self.in_deg / max(1.0, float(self.N - 1))
        x_scaled = self.x / (self.cfg.x_max + 1e-12)
        turnover_prev = float(np.mean(np.abs(self.delta_x_prev))) if self.t > 0 else 0.0

        # IMPORTANT: no per-step std normalization here (removed)
        obs = np.stack([
            np.full(self.N, self.p_log_return, dtype=np.float64),
            x_scaled.astype(np.float64),
            indeg_scaled.astype(np.float64),
            self.b.astype(np.float64),
            self.s.astype(np.float64),
            self.m.astype(np.float64),
            np.full(self.N, self.var_ewma, dtype=np.float64),
            np.full(self.N, self.last_price_ret, dtype=np.float64),
            np.full(self.N, self.last_R_port, dtype=np.float64),
            np.full(self.N, turnover_prev, dtype=np.float64),
        ], axis=1)
        return obs.astype(np.float32)


# ===================== ACTOR / CRITIC =====================
class Actor(nn.Module):
    def __init__(self, obs_dim: int, cfg: Config):
        super().__init__()
        h1, h2 = cfg.actor_hidden
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 2),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.net(x)
        alpha = torch.nn.functional.softplus(out[..., 0]) + 1.0
        beta = torch.nn.functional.softplus(out[..., 1]) + 1.0
        return alpha, beta

    def dist(self, x):
        a, b = self.forward(x)
        return Beta(a, b)

    def logp_entropy(self, x, act):
        d = self.dist(x)
        logp = d.log_prob(act)
        ent = d.entropy()
        return logp, ent


class Critic(nn.Module):
    def __init__(self, obs_dim: int, cfg: Config):
        super().__init__()
        h1, h2, h3 = cfg.critic_hidden
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, pooled_obs):
        return self.net(pooled_obs).squeeze(-1)


# ===================== PPO AGENT =====================
class PPOAgent:
    def __init__(self, obs_dim: int, cfg: Config, normalizer: Optional[RunningMeanStd] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(obs_dim, cfg).to(self.device)
        self.critic = Critic(obs_dim, cfg).to(self.device)

        self.optA = optim.Adam(self.actor.parameters(), lr=cfg.lrA)
        self.optV = optim.Adam(self.critic.parameters(), lr=cfg.lrV)

        self.norm = normalizer
        self.reset_buf()

    def reset_buf(self):
        self.obs = []
        self.act = []
        self.logp = []
        self.rew = []
        self.done = []
        self.next_obs = []

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.norm is None:
            return obs
        # update normalizer using current obs batch (flatten agents)
        flat = obs.reshape(-1, obs.shape[-1])
        self.norm.update(flat)
        return self.norm.normalize(obs)

    @torch.no_grad()
    def act_step(self, obs_np: np.ndarray, deterministic: bool = False):
        obs_np = self._normalize_obs(obs_np)

        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        d = self.actor.dist(x)

        if deterministic:
            a, b = self.actor.forward(x)
            act = a / (a + b)
            logp = torch.zeros_like(act)
        else:
            act = d.sample()
            logp = d.log_prob(act)

        return (
            act.detach().cpu().numpy().astype(np.float32),
            logp.detach().cpu().numpy().astype(np.float32),
        )

    def store(self, obs, act, logp, rew, done, next_obs):
        # Store RAW obs; normalize later in update using running stats (already updated online too)
        self.obs.append(obs.copy())
        self.act.append(act.copy())
        self.logp.append(logp.copy())
        self.rew.append(float(rew))
        self.done.append(float(done))
        self.next_obs.append(next_obs.copy())

    def _gae_with_vnext(self, v, v_next, r, d):
        T = len(r)
        adv = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - d[t]
            delta = r[t] + self.cfg.gamma * v_next[t] * mask - v[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            adv[t] = gae
        ret = adv + v
        return adv, ret

    def update(self):
        if len(self.obs) == 0:
            return

        cfg = self.cfg
        device = self.device

        obs = np.asarray(self.obs, dtype=np.float32)          # (T,N,D)
        act = np.asarray(self.act, dtype=np.float32)          # (T,N)
        logp_old = np.asarray(self.logp, dtype=np.float32)    # (T,N)
        rew = np.asarray(self.rew, dtype=np.float32)          # (T,)
        done = np.asarray(self.done, dtype=np.float32)        # (T,)
        next_obs = np.asarray(self.next_obs, dtype=np.float32)# (T,N,D)

        T, N, D = obs.shape

        # Normalize using the same normalizer (do NOT update stats here again to avoid leakage)
        if self.norm is not None:
            obs_n = self.norm.normalize(obs)
            next_obs_n = self.norm.normalize(next_obs)
        else:
            obs_n, next_obs_n = obs, next_obs

        pooled = obs_n.mean(axis=1)          # (T,D)
        pooled_next = next_obs_n.mean(axis=1)# (T,D)

        pooled_t = torch.tensor(pooled, dtype=torch.float32, device=device)
        pooled_next_t = torch.tensor(pooled_next, dtype=torch.float32, device=device)

        with torch.no_grad():
            v = self.critic(pooled_t).cpu().numpy().astype(np.float64)        # (T,)
            v_next = self.critic(pooled_next_t).cpu().numpy().astype(np.float64)  # (T,)

        adv, ret = self._gae_with_vnext(v, v_next, rew.astype(np.float64), done.astype(np.float64))

        if adv.std() > 1e-12:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            adv = adv * 0.0

        adv_rep = np.repeat(adv, N)  # (T*N,)

        obs_flat = obs_n.reshape(T * N, D)
        act_flat = act.reshape(T * N)
        logp_old_flat = logp_old.reshape(T * N)

        obs_flat_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
        act_flat_t = torch.tensor(act_flat, dtype=torch.float32, device=device)
        logp_old_flat_t = torch.tensor(logp_old_flat, dtype=torch.float32, device=device)
        adv_flat_t = torch.tensor(adv_rep, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        M = T * N
        batchA = min(cfg.batchA, M)
        batchV = min(cfg.batchV, T)

        for _ in range(cfg.ppo_epochs):
            # critic
            idxV = np.random.permutation(T)
            for start in range(0, T, batchV):
                mb = idxV[start:start + batchV]
                if mb.size == 0:
                    continue
                v_pred = self.critic(pooled_t[mb])
                v_loss = nn.MSELoss()(v_pred, ret_t[mb])

                self.optV.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optV.step()

            # actor
            idxA = np.random.permutation(M)
            for start in range(0, M, batchA):
                mb = idxA[start:start + batchA]
                if mb.size == 0:
                    continue

                logp_new, ent = self.actor.logp_entropy(obs_flat_t[mb], act_flat_t[mb])
                ratio = torch.exp(logp_new - logp_old_flat_t[mb])

                surr1 = ratio * adv_flat_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_flat_t[mb]
                pg_loss = -torch.min(surr1, surr2).mean()

                lossA = pg_loss - cfg.ent_coef * ent.mean()

                self.optA.zero_grad(set_to_none=True)
                lossA.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optA.step()

        self.reset_buf()


# ===================== EVALUATION =====================
def run_episode_fixed_gate(cfg: Config, g: float, resample_network: bool = False):
    env = NetworkMarketEnv(cfg, resample_each_episode=resample_network)
    env.reset()

    while True:
        actions = np.full(cfg.N, float(g), dtype=np.float32)
        _, _, done, _ = env.step(actions)
        if done:
            R = np.asarray(env.R_port_hist, dtype=np.float64)
            return {
                "gate": float(g),
                "Sharpe": sharpe_annualized(R),
                "MeanR": float(R.mean()) if R.size else 0.0,
                "StdR": float(R.std()) if R.size else 0.0,
                "CumReturn": calculate_cumulative_return(R),
                "CVaR": tail_metrics(R, alpha=cfg.cvar_alpha)["CVaR"],
                "FinalWealth": env.wealth,
                "Turn": float(np.mean(env.turn_hist)) if len(env.turn_hist) else 0.0,
            }


def evaluate_fixed_frontier(cfg: Config, resample_network: bool = False):
    return [run_episode_fixed_gate(cfg, float(g), resample_network) for g in cfg.fixed_gates]


def evaluate_policy_multiple_runs(cfg: Config, agent: PPOAgent,
                                 deterministic: bool = True,
                                 resample_network: bool = False,
                                 n_runs: int = 10):
    all_results = []
    for _ in range(n_runs):
        env = NetworkMarketEnv(cfg, resample_each_episode=resample_network)
        obs = env.reset()
        while True:
            a, _ = agent.act_step(obs, deterministic=deterministic)
            obs, _, done, _ = env.step(a)
            if done:
                R = np.asarray(env.R_port_hist, dtype=np.float64)
                all_results.append({
                    "Sharpe": sharpe_annualized(R),
                    "CumReturn": calculate_cumulative_return(R),
                    "CVaR": tail_metrics(R, alpha=cfg.cvar_alpha)["CVaR"],
                    "MeanGate": float(np.mean(env.gate_mean_hist)) if len(env.gate_mean_hist) else 0.0,
                    "Turn": float(np.mean(env.turn_hist)) if len(env.turn_hist) else 0.0,
                })
                break

    agg = {
        "Sharpe_mean": float(np.mean([r["Sharpe"] for r in all_results])),
        "Sharpe_std": float(np.std([r["Sharpe"] for r in all_results])),
        "CumReturn_mean": float(np.mean([r["CumReturn"] for r in all_results])),
        "CumReturn_std": float(np.std([r["CumReturn"] for r in all_results])),
        "CVaR_mean": float(np.mean([r["CVaR"] for r in all_results])),
        "MeanGate_mean": float(np.mean([r["MeanGate"] for r in all_results])),
        "MeanGate_std": float(np.std([r["MeanGate"] for r in all_results])),
        "Turn_mean": float(np.mean([r["Turn"] for r in all_results])),
        "AllRuns": all_results,
    }
    return agg


# ===================== TRAINING =====================
def train(cfg: Config, resample_each_episode: bool = False):
    env = NetworkMarketEnv(cfg, resample_each_episode=resample_each_episode)
    obs_dim = env.reset().shape[1]

    # running normalizer for obs_dim
    rms = RunningMeanStd(shape=(obs_dim,))
    agent = PPOAgent(obs_dim, cfg, normalizer=rms)

    hist = {"episode_rewards": [], "episode_sharpes": [], "episode_gates": [], "episode_wealth": []}

    print("\nStarting training...")
    print(f"Episodes: {cfg.episodes}")
    print("-" * 100)

    for ep in range(cfg.episodes):
        obs = env.reset()
        ep_rew = 0.0

        while True:
            act, logp = agent.act_step(obs, deterministic=False)
            next_obs, reward, done, info = env.step(act)

            agent.store(obs, act, logp, reward, done, next_obs)
            ep_rew += reward
            obs = next_obs

            if done:
                agent.update()

                R = np.asarray(env.R_port_hist, dtype=np.float64)
                sh = sharpe_annualized(R)

                hist["episode_rewards"].append(ep_rew)
                hist["episode_sharpes"].append(sh)
                hist["episode_gates"].append(info["mean_gate"])
                hist["episode_wealth"].append(info["wealth"])

                if info["mean_gate"] < cfg.trivial_gate_threshold and ep % 10 == 0:
                    print(f"WARNING: Trivial-ish gate (mean_gate={info['mean_gate']:.3f})")

                if ep % 10 == 0:
                    tm = tail_metrics(R, alpha=cfg.cvar_alpha)
                    print(
                        f"Ep {ep:04d} | "
                        f"Reward: {ep_rew:+.4f} | "
                        f"Gate: {info['mean_gate']:.3f} | "
                        f"Sharpe: {sh:+.3f} | "
                        f"Wealth: {info['wealth']:.2f} | "
                        f"CVaR: {tm['CVaR']:+.6f}"
                    )
                break

    return agent, hist


# ===================== MAIN =====================
def main():
    set_seed(CFG.seed)

    print("=" * 100)
    print("PAPER1: Gate-only PPO for Network Market (FIXED)")
    print("=" * 100)
    print(f"N={CFG.N}, K={CFG.K}, T={CFG.T}, Episodes={CFG.episodes}, device={CFG.device}")
    print("=" * 100)

    # 1) Fixed-gate frontier
    print("\n1) Fixed Gate Frontier...")
    frontier = evaluate_fixed_frontier(CFG, resample_network=False)
    print("Gate | Sharpe  | CumReturn | CVaR      | Turn    | Wealth")
    print("-" * 85)
    best = None
    for r in frontier:
        best = r if (best is None or r["Sharpe"] > best["Sharpe"]) else best
        print(f"{r['gate']:4.1f} | {r['Sharpe']:7.3f} | {r['CumReturn']:9.3%} | {r['CVaR']:+.6f} | "
              f"{r['Turn']:7.4f} | {r['FinalWealth']:7.2f}")

    if best:
        print(f"\nBest fixed gate by Sharpe: g={best['gate']:.2f}, Sharpe={best['Sharpe']:+.3f}")

    # 2) Train
    print("\n2) Training PPO...")
    t0 = time.time()
    agent, hist = train(CFG, resample_each_episode=False)
    dt = time.time() - t0
    print(f"\nTraining done in {dt:.2f}s")

    # 3) Eval multiple runs
    print("\n3) Evaluating learned policy...")
    eval_res = evaluate_policy_multiple_runs(CFG, agent, deterministic=True, resample_network=False, n_runs=CFG.eval_runs)
    print(f"Sharpe: {eval_res['Sharpe_mean']:+.3f} ± {eval_res['Sharpe_std']:.3f}")
    print(f"CumReturn: {eval_res['CumReturn_mean']:+.3%} ± {eval_res['CumReturn_std']:.3%}")
    print(f"MeanGate: {eval_res['MeanGate_mean']:.3f} ± {eval_res['MeanGate_std']:.3f}")
    print(f"CVaR: {eval_res['CVaR_mean']:+.6f}")
    print(f"Turn: {eval_res['Turn_mean']:.4f}")


if __name__ == "__main__":
    main()
