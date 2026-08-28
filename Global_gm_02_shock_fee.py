# -*- coding: utf-8 -*-
"""
Paper1 | Global g_t + Band-triggered q_t + CB inventory(cap-safe) + Fee Fund + SHOCK | PPO
--------------------------------------------------------------------------------
Adds:
- Fundamental shock at time t_shock (random sign per episode by default)
- Optional: use distortion penalty in relative units (d^2) instead of (p-p_f)^2
"""

import time
from dataclasses import dataclass
from typing import Tuple, Dict

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

    # Price
    p0: float = 100.0
    alpha_fund: float = 0.01
    kappa: float = 0.02
    sigma_eps: float = 0.01

    # Followers BR / inventory
    a_max: float = 0.50
    x_cap: float = 20.0              # hard cap on follower inventory (keeps model stable)

    lam0: float = 0.10
    lam1: float = 0.40
    tau0: float = 0.01               # base soft-threshold cost in BR (units of mu)
    eta_inv: float = 0.05            # holding/financing pressure (inventory pullback)
    mu_scale: float = 1.0

    # Transaction fee (goes to Fund)
    tau_fee: float = 0.001           # 10bp
    # NOTE: fee_i,t = tau_fee * p_t * |a_i,t|

    # Risk EWMA
    beta_ewma: float = 0.95

    # Band-trigger q_t (CB intervention)
    delta: float = 0.02              # 2% distortion band
    d_clip: float = 0.10             # clip distortion used inside q law (10%)
    k_q: float = 50.0                # intervention slope on distortion (tune)
    q_max: float = 2.0               # max q per step
    Xcb_cap: float = 200.0           # CB inventory cap

    # MM reward weights
    w_dist: float = 1.0
    w_var: float = 0.50
    w_imp: float = 1.0
    w_liq: float = 1.0
    w_inv: float = 0.25
    w_q: float = 0.50
    w_xcb: float = 0.10

    # Liquidity band (penalize low activity)
    L_min: float = 0.08
    liq_band_power: float = 2.0

    # ===== SHOCK SETTINGS =====
    shock_on: bool = True
    shock_t: int = 250                 # when the shock hits
    shock_size_y: float = 5.0          # additive jump to y (big enough to matter)
    shock_random_sign: bool = True     # random +/-
    shock_sign: float = +1.0           # used only if shock_random_sign=False
    shock_once: bool = True            # apply at most once per episode

    # Optional: better scaling
    use_relative_dist_penalty: bool = True  # if True, use d^2 instead of (p-p_f)^2

    # PPO
    lrA: float = 3e-4
    lrV: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    ppo_epochs: int = 8
    batchA: int = 2048
    batchV: int = 128

    # Networks
    actor_hidden: Tuple[int, int] = (128, 64)
    critic_hidden: Tuple[int, int, int] = (128, 64, 32)

    # Debug
    debug_first_steps: int = 5

    # Random / device
    seed: int = 42
    device: str = "cpu"


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


def sharpe_annualized(r: np.ndarray, scale: float = 252.0):
    r = np.asarray(r, dtype=np.float64)
    if r.size < 2:
        return 0.0
    s = r.std()
    if s <= 1e-12:
        return 0.0
    return float(r.mean() / s * np.sqrt(scale))


# ===================== ENV =====================
class GlobalGateBandCBEnv:
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

    @staticmethod
    def _soft_threshold(x: np.ndarray, tau: float):
        ax = np.abs(x)
        return np.sign(x) * np.maximum(ax - tau, 0.0)

    def reset(self):
        self._maybe_build_network()

        self.t = 0
        self.y = float(self.cfg.y0)
        self.p = float(self.cfg.p0)
        self.p_fund = float(self.cfg.p0)

        # shock per episode
        self.shock_applied = False
        if self.cfg.shock_on and self.cfg.shock_random_sign:
            self.shock_dir = float(self.rng.choice([-1.0, +1.0]))
        else:
            self.shock_dir = float(self.cfg.shock_sign)

        # beliefs
        self.b = self.rng.normal(0.0, 0.1, self.N).astype(np.float64)
        self.s = np.zeros(self.N, dtype=np.float64)
        self.m = (self.P @ self.b).astype(np.float64)

        # follower inventory
        self.x = np.zeros(self.N, dtype=np.float64)

        # CB
        self.Xcb = 0.0
        self.fund = 0.0

        # risk tracking
        self.var_ewma = 1e-6
        self.last_price_ret = 0.0

        # histories (for reporting)
        self.price_ret_hist = []
        self.g_hist = []
        self.sum_a_hist = []
        self.mean_abs_a_hist = []
        self.q_hist = []
        self.Xcb_hist = []
        self.d_hist = []
        self.mm_rew_hist = []

        # init
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
        self.b = ((1.0 - self.cfg.omega) * self.s + self.cfg.omega * (self.P @ self.b) + noise_b).astype(np.float64)
        self.m = (self.P @ self.b).astype(np.float64)

    def followers_best_response(self, g: float) -> np.ndarray:
        p = float(self.p)
        mu = self.cfg.mu_scale * (self.cfg.alpha_fund * p) * self.m
        mu_eff = (g * mu) - self.cfg.eta_inv * self.x

        tau_eff = self.cfg.tau0 + self.cfg.tau_fee * p
        numer = self._soft_threshold(mu_eff, tau_eff)
        denom = 2.0 * (self.cfg.lam0 + self.cfg.lam1 * g + 1e-12)

        a = numer / denom
        a = np.clip(a, -self.cfg.a_max, self.cfg.a_max)
        return a.astype(np.float64)

    def _compute_q_cap_safe(self, d_prev: float) -> Dict[str, float]:
        cfg = self.cfg
        I = 1.0 if abs(d_prev) > cfg.delta else 0.0
        d_used = float(np.clip(d_prev, -cfg.d_clip, cfg.d_clip))
        q_raw = -I * cfg.k_q * d_used
        q_raw = float(np.clip(q_raw, -cfg.q_max, cfg.q_max))

        q_applied = float(np.clip(q_raw, -cfg.Xcb_cap - self.Xcb, cfg.Xcb_cap - self.Xcb))
        return {"I": I, "q_raw": q_raw, "q": q_applied}

    def _maybe_apply_shock(self):
        cfg = self.cfg
        if not cfg.shock_on:
            return
        if self.t != cfg.shock_t:
            return
        if cfg.shock_once and self.shock_applied:
            return

        # Fundamental news shock: jump y
        self.y = float(self.y + self.shock_dir * cfg.shock_size_y)
        self.shock_applied = True

    def step(self, g_action: float, debug: bool = False):
        cfg = self.cfg
        g = float(np.clip(float(g_action), 0.0, 1.0))

        # Apply shock at the START of step (so it hits pricing this step)
        self._maybe_apply_shock()

        p_prev = float(self.p)
        p_f_prev = float(self.p_fund)

        d_prev = float((p_prev - p_f_prev) / (p_f_prev + 1e-12))

        a = self.followers_best_response(g)
        sum_a = float(a.sum())
        mean_abs_a = float(np.mean(np.abs(a)))

        fee_total = float(cfg.tau_fee * p_prev * np.sum(np.abs(a)))
        self.fund += fee_total

        q_pack = self._compute_q_cap_safe(d_prev)
        q = q_pack["q"]
        self.Xcb = float(self.Xcb + q)

        fund_move = cfg.alpha_fund * self.y * p_prev
        eps = float(self.rng.normal(0.0, cfg.sigma_eps))
        impact = cfg.kappa * (sum_a + q)

        p_new = p_prev + fund_move + impact + eps
        p_new = max(p_new, 0.1)

        p_f_new = p_f_prev + fund_move

        self.x = np.clip(self.x + a, -cfg.x_cap, cfg.x_cap).astype(np.float64)

        price_ret = float((p_new - p_prev) / (p_prev + 1e-12))
        self.last_price_ret = price_ret
        self.var_ewma = float(cfg.beta_ewma * self.var_ewma + (1.0 - cfg.beta_ewma) * (price_ret ** 2))

        d_new = float((p_new - p_f_new) / (p_f_new + 1e-12))

        liq_gap = max(0.0, cfg.L_min - mean_abs_a)
        liq_pen = float(liq_gap ** cfg.liq_band_power)

        inv_pen = float(np.mean(self.x ** 2))
        imp2 = float((impact) ** 2)
        q2 = float(q ** 2)
        Xcb2 = float(self.Xcb ** 2)

        # Distortion penalty: choose one
        if cfg.use_relative_dist_penalty:
            dist_term = float(d_new ** 2)  # consistent with state
        else:
            dist_term = float((p_new - p_f_new) ** 2)

        mm_loss = (
            cfg.w_dist * dist_term
            + cfg.w_var * self.var_ewma
            + cfg.w_imp * imp2
            + cfg.w_liq * liq_pen
            + cfg.w_inv * inv_pen
            + cfg.w_q * q2
            + cfg.w_xcb * Xcb2
        )
        reward = -mm_loss

        self.p = float(p_new)
        self.p_fund = float(p_f_new)

        # Next signals/beliefs
        self._update_fundamental_and_signals()
        self._update_beliefs()

        self.price_ret_hist.append(price_ret)
        self.g_hist.append(g)
        self.sum_a_hist.append(sum_a)
        self.mean_abs_a_hist.append(mean_abs_a)
        self.q_hist.append(q)
        self.Xcb_hist.append(self.Xcb)
        self.d_hist.append(d_new)
        self.mm_rew_hist.append(reward)

        if debug:
            shock_flag = 1 if (cfg.shock_on and self.shock_applied and self.t == cfg.shock_t) else 0
            print(
                f"[t={self.t:04d}] g={g:.3f} shock={shock_flag} | d_prev={d_prev:+.4%} I={int(q_pack['I'])} "
                f"q={q:+.3f} Xcb={self.Xcb:+.2f} | sum_a={sum_a:+.3f} mean|a|={mean_abs_a:.3f} "
                f"feeTot={fee_total:.3f} fund={self.fund:.2f} | impact={impact:+.4f} "
                f"p={p_new:.3f} p_f={p_f_new:.3f} d_new={d_new:+.4%} | loss={mm_loss:.6f} rew={reward:.6f}"
            )

        self.t += 1
        done = (self.t >= cfg.T)

        info = {
            "g": g,
            "d_prev": d_prev,
            "d_new": d_new,
            "q": q,
            "Xcb": self.Xcb,
            "sum_a": sum_a,
            "mean_abs_a": mean_abs_a,
            "fee_total": fee_total,
            "fund": self.fund,
            "var": self.var_ewma,
            "mm_loss": mm_loss,
            "p": self.p,
            "p_fund": self.p_fund,
            "shock_applied": self.shock_applied,
            "shock_dir": self.shock_dir,
        }

        return self._get_obs(), float(reward), done, info

    def _get_obs(self):
        cfg = self.cfg
        if len(self.d_hist) > 0:
            d = float(self.d_hist[-1])
        else:
            d = float((self.p - self.p_fund) / (self.p_fund + 1e-12))

        mean_abs_a_prev = float(self.mean_abs_a_hist[-1]) if len(self.mean_abs_a_hist) else 0.0
        I = 1.0 if abs(d) > cfg.delta else 0.0

        obs = np.array([
            abs(d),
            float(self.var_ewma),
            mean_abs_a_prev,
            I,
            abs(float(self.Xcb)) / (cfg.Xcb_cap + 1e-12),
        ], dtype=np.float32)

        return obs


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
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.net(x)
        a = torch.nn.functional.softplus(out[..., 0]) + 1.0
        b = torch.nn.functional.softplus(out[..., 1]) + 1.0
        return a, b

    def dist(self, x):
        a, b = self.forward(x)
        return Beta(a, b)


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
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===================== PPO AGENT =====================
class PPOAgent:
    def __init__(self, obs_dim: int, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(obs_dim, cfg).to(self.device)
        self.critic = Critic(obs_dim, cfg).to(self.device)

        self.optA = optim.Adam(self.actor.parameters(), lr=cfg.lrA)
        self.optV = optim.Adam(self.critic.parameters(), lr=cfg.lrV)

        self.reset_buf()

    def reset_buf(self):
        self.obs = []
        self.act = []
        self.logp = []
        self.rew = []
        self.done = []
        self.next_obs = []

    @torch.no_grad()
    def act_step(self, obs_np: np.ndarray, deterministic: bool = False):
        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        d = self.actor.dist(x)

        if deterministic:
            a, b = self.actor.forward(x)
            act = a / (a + b)
            logp = torch.zeros_like(act)
        else:
            act = d.sample()
            logp = d.log_prob(act)

        return float(act.item()), float(logp.item())

    def store(self, obs, act, logp, rew, done, next_obs):
        self.obs.append(obs.copy())
        self.act.append(float(act))
        self.logp.append(float(logp))
        self.rew.append(float(rew))
        self.done.append(float(done))
        self.next_obs.append(next_obs.copy())

    def _gae(self, v, r, d, last_v):
        T = len(r)
        adv = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - d[t]
            next_v = last_v if t == T - 1 else v[t + 1]
            delta = r[t] + self.cfg.gamma * next_v * mask - v[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            adv[t] = gae
        ret = adv + v
        return adv, ret

    def update(self):
        if len(self.obs) == 0:
            return

        cfg = self.cfg
        device = self.device

        obs = np.asarray(self.obs, dtype=np.float32)
        act = np.asarray(self.act, dtype=np.float32)
        logp_old = np.asarray(self.logp, dtype=np.float32)
        rew = np.asarray(self.rew, dtype=np.float32)
        done = np.asarray(self.done, dtype=np.float32)
        next_obs = np.asarray(self.next_obs, dtype=np.float32)

        T, D = obs.shape

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            v = self.critic(obs_t).cpu().numpy().astype(np.float64)
            last_v = float(self.critic(next_obs_t[-1:]).item()) * (1.0 - float(done[-1]))

        adv, ret = self._gae(v, rew.astype(np.float64), done.astype(np.float64), last_v)

        if adv.std() > 1e-12:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            adv = adv * 0.0

        act_t = torch.tensor(act, dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(logp_old, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        batchA = min(cfg.batchA, T)
        batchV = min(cfg.batchV, T)

        for _ in range(cfg.ppo_epochs):
            idxV = np.random.permutation(T)
            for start in range(0, T, batchV):
                mb = idxV[start:start + batchV]
                if mb.size == 0:
                    continue
                v_pred = self.critic(obs_t[mb])
                v_loss = nn.MSELoss()(v_pred, ret_t[mb])
                self.optV.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optV.step()

            idxA = np.random.permutation(T)
            for start in range(0, T, batchA):
                mb = idxA[start:start + batchA]
                if mb.size == 0:
                    continue

                d = self.actor.dist(obs_t[mb])
                logp_new = d.log_prob(act_t[mb])
                ent = d.entropy()

                ratio = torch.exp(logp_new - logp_old_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_t[mb]
                pg_loss = -torch.min(s1, s2).mean()
                lossA = pg_loss - cfg.ent_coef * ent.mean()

                self.optA.zero_grad(set_to_none=True)
                lossA.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optA.step()

        self.reset_buf()


# ===================== BASELINES / TRAIN / EVAL =====================
def run_fixed_gate(cfg: Config, g: float, debug_steps: int = 0):
    env = GlobalGateBandCBEnv(cfg, resample_each_episode=False)
    obs = env.reset()
    ep_rew = 0.0

    for t in range(cfg.T):
        obs, rew, done, info = env.step(g, debug=(t < debug_steps))
        ep_rew += rew
        if done:
            break

    R = np.asarray(env.price_ret_hist, dtype=np.float64)
    return {
        "g": float(g),
        "ep_rew": float(ep_rew),
        "Sharpe": sharpe_annualized(R),
        "d_rms": float(np.sqrt(np.mean(np.square(env.d_hist)))) if len(env.d_hist) else 0.0,
        "mean_abs_a": float(np.mean(env.mean_abs_a_hist)) if len(env.mean_abs_a_hist) else 0.0,
        "mean_sum_a": float(np.mean(env.sum_a_hist)) if len(env.sum_a_hist) else 0.0,
        "mean_q": float(np.mean(env.q_hist)) if len(env.q_hist) else 0.0,
        "Xcb_end": float(env.Xcb),
        "Fund_end": float(env.fund),
    }


def train_ppo(cfg: Config):
    env = GlobalGateBandCBEnv(cfg, resample_each_episode=False)
    obs_dim = env.reset().shape[0]
    agent = PPOAgent(obs_dim, cfg)

    print("\n=== Train PPO MM (global g + band q + cap-safe Xcb + fee fund + shock) ===")
    for ep in range(cfg.episodes):
        obs = env.reset()
        ep_rew = 0.0

        for t in range(cfg.T):
            g, logp = agent.act_step(obs, deterministic=False)
            next_obs, rew, done, info = env.step(g, debug=False)

            agent.store(obs, g, logp, rew, done, next_obs)
            ep_rew += rew
            obs = next_obs

            if done:
                agent.update()

                if ep % 10 == 0:
                    R = np.asarray(env.price_ret_hist, dtype=np.float64)
                    print(
                        f"ep={ep:04d} | ep_rew={ep_rew:+.6f} | g_mean={np.mean(env.g_hist):.3f} | "
                        f"d_rms={np.sqrt(np.mean(np.square(env.d_hist))):.4%} | var={env.var_ewma:.3e} | "
                        f"mean|a|={np.mean(env.mean_abs_a_hist):.3f} | mean q={np.mean(env.q_hist):+.3f} | "
                        f"Xcb_end={env.Xcb:+.2f} | Fund_end={env.fund:.2f} | Sharpe={sharpe_annualized(R):+.3f} | "
                        f"shock_dir={env.shock_dir:+.0f}"
                    )
                break

    return agent


def eval_learned(cfg: Config, agent: PPOAgent, deterministic: bool = True, debug_steps: int = 5):
    env = GlobalGateBandCBEnv(cfg, resample_each_episode=False)
    obs = env.reset()
    ep_rew = 0.0

    for t in range(cfg.T):
        g, _ = agent.act_step(obs, deterministic=deterministic)
        obs, rew, done, info = env.step(g, debug=(t < debug_steps or t == cfg.shock_t))
        ep_rew += rew
        if done:
            break

    R = np.asarray(env.price_ret_hist, dtype=np.float64)
    return {
        "ep_rew": float(ep_rew),
        "g_mean": float(np.mean(env.g_hist)) if len(env.g_hist) else float(info["g"]),
        "Sharpe": sharpe_annualized(R),
        "d_rms": float(np.sqrt(np.mean(np.square(env.d_hist)))) if len(env.d_hist) else 0.0,
        "mean_abs_a": float(np.mean(env.mean_abs_a_hist)) if len(env.mean_abs_a_hist) else 0.0,
        "mean_sum_a": float(np.mean(env.sum_a_hist)) if len(env.sum_a_hist) else 0.0,
        "mean_q": float(np.mean(env.q_hist)) if len(env.q_hist) else 0.0,
        "Xcb_end": float(env.Xcb),
        "Fund_end": float(env.fund),
        "shock_dir": float(env.shock_dir),
    }


# ===================== MAIN =====================
def main():
    set_seed(CFG.seed)

    print("=" * 120)
    print("Paper1 | Global g_t + Band q_t + CAP-SAFE CB inventory + Transaction Fee Fund + SHOCK | PPO")
    print("=" * 120)
    print(f"N={CFG.N}, K={CFG.K}, T={CFG.T}, episodes={CFG.episodes}, device={CFG.device}")
    print(f"Band: delta={CFG.delta:.2%}, q_max={CFG.q_max}, Xcb_cap={CFG.Xcb_cap}")
    print(f"Fee: tau_fee={CFG.tau_fee} -> fee_total_t = tau_fee * p_t * sum|a_i| goes to FUND")
    print(f"Shock: on={CFG.shock_on} at t={CFG.shock_t}, size_y={CFG.shock_size_y}, random_sign={CFG.shock_random_sign}")
    print(f"Dist penalty: relative(d^2)={CFG.use_relative_dist_penalty}")
    print("=" * 120)

    print("\n=== 1) Fixed-gate baselines ===")
    grid = np.linspace(0.0, 1.0, 11)
    print(" g  | ep_rew | Sharpe | d_rms   | mean|a| | mean(sum_a) | mean(q) | Xcb_end | Fund_end")
    print("-" * 120)
    best = None
    for g in grid:
        dbg = CFG.debug_first_steps if (abs(g - 0.0) < 1e-12 or abs(g - 1.0) < 1e-12) else 0
        r = run_fixed_gate(CFG, float(g), debug_steps=dbg)
        if best is None or r["ep_rew"] > best["ep_rew"]:
            best = r
        print(
            f"{r['g']:.1f} | {r['ep_rew']:+10.4f} | {r['Sharpe']:+7.3f} | {r['d_rms']:7.3%} | "
            f"{r['mean_abs_a']:7.3f} | {r['mean_sum_a']:+11.3f} | {r['mean_q']:+7.3f} | "
            f"{r['Xcb_end']:+7.2f} | {r['Fund_end']:8.2f}"
        )

    print(f"\nBest fixed g by ep_rew: g={best['g']:.2f} ep_rew={best['ep_rew']:+.6f}")

    print("\n=== 2) Train PPO (MM) ===")
    t0 = time.time()
    agent = train_ppo(CFG)
    print(f"Training done in {time.time() - t0:.1f}s")

    print("\n=== 3) Learned policy (deterministic) ===")
    out = eval_learned(CFG, agent, deterministic=True, debug_steps=CFG.debug_first_steps)
    print(
        f"ep_rew={out['ep_rew']:+.6f} | g_mean={out['g_mean']:.3f} | Sharpe={out['Sharpe']:+.3f} | "
        f"d_rms={out['d_rms']:.3%} | mean|a|={out['mean_abs_a']:.3f} | mean(sum_a)={out['mean_sum_a']:+.3f} | "
        f"mean(q)={out['mean_q']:+.3f} | Xcb_end={out['Xcb_end']:+.2f} | Fund_end={out['Fund_end']:.2f} | "
        f"shock_dir={out['shock_dir']:+.0f}"
    )


if __name__ == "__main__":
    main()
