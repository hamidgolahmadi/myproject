# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 12:29:04 2025

@author: hg2e25
"""

# -*- coding: utf-8 -*-
"""
ppo_network_only_v10_1.py  (v10.1)

Applies 4 critical fixes on v10:

(1) PPO aggressiveness reduced (KL control):
    - lr_actor lower, clip_range tighter, mini_epochs fewer

(2) Entropy bonus: use GATE entropy only (net entropy excluded to avoid scale mismatch)

(3) Info-task timing fix (no leakage):
    - pred_err_t = ( bbar_{t-1} - R_t )^2
    - bbar_{t-1} captured BEFORE this-step belief update and price/return realization

(4) Gate std clamp:
    - logstd in [LOGSTD_MIN, LOGSTD_MAX] (default [-2.5, 0.0])

Also:
- Gate is stochastic sigmoid-Gaussian; logprob included in PPO ratio every step.
- Network rewiring logprob included only when netmask==1.
- Prints PPO diagnostics + gate distribution stats.
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


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().data.norm(2).item()
        total += g * g
    return float(math.sqrt(total))


# =============================================================================
# Environment (Info network + rule-based finance with learned gate)
# =============================================================================

class InfoNetworkBondEnvV10:
    """
    Multi-agent env:
      - Information network P (row-stochastic, each i listens to K nodes)
      - Beliefs update: private signal + DeGroot mixing
      - Price impacted by net flow + noise
      - Action:
          * network rewiring every net_period (neighbors + weights)
          * gate g_i(t) in [0,1] scales rule-based action on network signal
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
        # risk EWMA
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

        # attached by runner:
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

        # IMPORTANT for info-task timing: store previous aggregate belief
        self.b_bar_prev = float(np.mean(self.b))

        return self._build_obs()

    def step(self, g_gate, neighbors=None, w=None):
        """
        Fix (3): pred_err timing is now correct:
            pred_err_t = (bbar_prev - R_t)^2
        where bbar_prev is the aggregate belief from t-1 (stored at end of last step).
        """

        # -----------------------------
        # Apply network action (slow timescale)
        # -----------------------------
        if (self.t % self.net_period) == 0 and neighbors is not None and w is not None:
            self.P_prev = self.P.copy()
            self.P = row_stochastic_from_neighbors(self.N, neighbors, w)

        # -----------------------------
        # Capture bbar_{t-1} BEFORE updating beliefs (no leakage)
        # -----------------------------
        bbar_prev_for_pred = float(self.b_bar_prev)

        # -----------------------------
        # Latent fundamental evolves + private signals
        # -----------------------------
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        # -----------------------------
        # Belief update (private + social)
        # -----------------------------
        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        # Update b_bar_prev for NEXT step's prediction target
        self.b_bar_prev = float(np.mean(self.b))

        # -----------------------------
        # Financial action (rule-based with learned gate)
        # -----------------------------
        g_gate = np.clip(np.asarray(g_gate, dtype=float), 0.0, 1.0)
        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * g_gate * signal)

        delta_x = a_fin * self.x_max
        tc = self.tau * np.abs(delta_x)
        self.x = self.x + delta_x

        # -----------------------------
        # Price formation (impact + noise)
        # -----------------------------
        net_flow = float(np.sum(delta_x))
        eps_p = float(self.rng.normal(0.0, self.sigma_eps))

        self.p_prev = self.p
        self.p = self.p + self.kappa * net_flow + eps_p
        self.p = max(self.p, 1e-6)

        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        # Risk state (EWMA R^2)
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

        # Fix (3): true predictive error (no same-step leakage)
        pred_err = float((bbar_prev_for_pred - R) ** 2)

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

LOGSTD_MIN = -2.5  # Fix (4)
LOGSTD_MAX =  0.0  # Fix (4)
EPS = 1e-8

class ActorNet(nn.Module):
    """
    Produces:
      - neighbor logits (N,N) from embeddings
      - weight logits (N,K) -> softmax weights (deterministic)
      - gate distribution params: mu_u (N,), logstd_u (N,) for u ~ N(mu, std)
        then gate g = sigmoid(u) in (0,1)
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

        mu_u = self.to_gate_mu(h).squeeze(-1)          # (N,)
        logstd_u = self.to_gate_logstd(h).squeeze(-1)  # (N,)
        logstd_u = torch.clamp(logstd_u, LOGSTD_MIN, LOGSTD_MAX)  # Fix (4)

        return logits, w_logits, mu_u, logstd_u


class CriticNet(nn.Module):
    """Central critic on mean-pooled obs."""
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
# Network sampling (masked sequential with indegree cap)
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
                # fallback
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
# Gate distribution: sigmoid-Gaussian with correct logprob
# =============================================================================

def sample_gate_and_logprob(mu_u: torch.Tensor, logstd_u: torch.Tensor):
    """
    u ~ Normal(mu, std); g = sigmoid(u)
    log p(g) = log p(u) - log |dg/du|  , where dg/du = sigmoid(u)(1-sigmoid(u))
    We store logp per-agent and sum/mean as needed.
    """
    std = torch.exp(logstd_u)
    dist = torch.distributions.Normal(mu_u, std)
    u = dist.rsample()
    g = torch.sigmoid(u)

    # Jacobian correction: log(dg/du) = log(g(1-g))
    log_det = torch.log(g * (1.0 - g) + EPS)
    logp_u = dist.log_prob(u)
    logp_g = logp_u - log_det

    # Entropy bonus: use entropy of u (stable proxy)  -- Fix (2): gate entropy only
    ent_u = dist.entropy()

    return g, u, logp_g, ent_u


def gate_logprob_from_u(mu_u: torch.Tensor, logstd_u: torch.Tensor, u: torch.Tensor):
    std = torch.exp(logstd_u)
    dist = torch.distributions.Normal(mu_u, std)
    g = torch.sigmoid(u)
    log_det = torch.log(g * (1.0 - g) + EPS)
    logp_u = dist.log_prob(u)
    logp_g = logp_u - log_det
    ent_u = dist.entropy()
    return g, logp_g, ent_u


# =============================================================================
# PPO config + GAE
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
    mini_epochs: int = 2     # Fix (1)
    minibatch: int = 64

    lr_actor: float = 5e-5   # Fix (1)
    lr_critic: float = 5e-4

    clip_range: float = 0.1  # Fix (1)
    target_kl: float = 0.02

    gae_lambda: float = 0.95
    discount: float = 0.99

    entropy_coef_gate: float = 0.01  # Fix (2): only gate entropy
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


# =============================================================================
# Rollout
# =============================================================================

@torch.no_grad()
def rollout_one(env: InfoNetworkBondEnvV10, actor: ActorNet, critic: CriticNet, cfg: PPOConfig):
    device = cfg.device

    obs = env.reset()
    obs_dim = len(obs[0])

    obs_list = []
    v_list = []
    r_list = []
    done_list = []

    # logprobs (total): gate always + network only when netmask==1
    logp_total_list = []
    logp_gate_list = []
    logp_net_list = []
    ent_gate_list = []

    netmask_list = []
    neighbors_list = []
    w_list = []
    u_list = []  # store gate pre-sigmoid sample u(t) to recompute logprob precisely

    dbg_last = {}

    for _ in range(cfg.rollout_len):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)  # (N,obs_dim)

        logits, w_logits, mu_u, logstd_u = actor(obs_t)

        # gate sample
        g, u, logp_gate_per_agent, ent_u_per_agent = sample_gate_and_logprob(mu_u, logstd_u)
        # aggregate gate logprob per step (mean over agents)
        logp_gate = logp_gate_per_agent.sum()
        ent_gate = ent_u_per_agent.mean()

        # network action maybe
        do_net = (env.t % cfg.net_period) == 0
        if do_net:
            neighbors, logp_net = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
            w = torch.softmax(w_logits, dim=-1)
            logp_net_scalar = logp_net
            neighbors_np = neighbors.cpu().numpy()
            w_np = w.cpu().numpy()
        else:
            logp_net_scalar = torch.zeros((), device=device)
            neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)
            neighbors_np, w_np = None, None

        # total logp per step
        logp_total = logp_gate + (logp_net_scalar if do_net else 0.0)

        # env step
        obs, r, done, info = env.step(
            g_gate=g.cpu().numpy(),
            neighbors=neighbors_np,
            w=w_np,
        )

        # critic value for state
        v = critic(obs_t).detach()

        # store
        obs_list.append(obs_t.detach())
        v_list.append(v)
        r_list.append(float(r))
        done_list.append(float(done))

        logp_total_list.append(logp_total.detach())
        logp_gate_list.append(logp_gate.detach())
        logp_net_list.append(logp_net_scalar.detach())
        ent_gate_list.append(ent_gate.detach())

        netmask_list.append(float(do_net))
        neighbors_list.append(neighbors.detach())
        w_list.append(w.detach())
        u_list.append(u.detach())

        dbg_last = info

        if done:
            break

    batch = {
        "obs": torch.stack(obs_list),  # (T,N,obs_dim)
        "v": torch.stack(v_list).squeeze(-1),  # (T,)
        "rew": torch.tensor(r_list, dtype=torch.float32, device=device),
        "done": torch.tensor(done_list, dtype=torch.float32, device=device),
        "logp_total": torch.stack(logp_total_list),  # (T,)
        "logp_gate": torch.stack(logp_gate_list),
        "logp_net": torch.stack(logp_net_list),
        "ent_gate": torch.stack(ent_gate_list),
        "netmask": torch.tensor(netmask_list, dtype=torch.float32, device=device),
        "neighbors": neighbors_list,
        "w": w_list,
        "u": u_list,
        "dbg_last": dbg_last,
        "obs_dim": obs_dim,
    }
    return batch


# =============================================================================
# Train v10.1
# =============================================================================

def train_v10_1():
    cfg = PPOConfig(
        horizon=1000,
        gamma_fin=5.0,
        net_period=5,
        K=5,
        indeg_cap=8,
        n_iters=200,
        rollout_len=200,
        mini_epochs=2,     # Fix (1)
        minibatch=64,
        lr_actor=5e-5,     # Fix (1)
        lr_critic=5e-4,
        clip_range=0.1,    # Fix (1)
        target_kl=0.02,
        gae_lambda=0.95,
        discount=0.99,
        entropy_coef_gate=0.01,  # Fix (2)
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
        w_info=2.0,  # info-task
    )

    print("=" * 110)
    print(
        f"CONFIG v10.1: horizon={cfg.horizon} | gamma_fin={cfg.gamma_fin} | K={cfg.K} | "
        f"indeg_cap={cfg.indeg_cap} | net_period={cfg.net_period}"
    )
    print(
        f"Reward: w_risk={env_kwargs['w_risk']} w_flow2={env_kwargs['w_flow2']} w_flow4={env_kwargs['w_flow4']} "
        f"w_gini={env_kwargs['w_gini']} w_net={env_kwargs['w_net']} w_info={env_kwargs['w_info']}"
    )
    print(
        f"PPO: lr_actor={cfg.lr_actor} clip_range={cfg.clip_range} mini_epochs={cfg.mini_epochs} "
        f"| Gate logstd in [{LOGSTD_MIN}, {LOGSTD_MAX}] | Entropy bonus: gate-only"
    )
    print("=" * 110)

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

    for it in range(cfg.n_iters):
        batch = rollout_one(env, actor, critic, cfg)

        obs = batch["obs"]
        T = obs.shape[0]

        # values + bootstrap
        with torch.no_grad():
            v = batch["v"]
            v_next = critic(obs[-1]) * (1.0 - batch["done"][-1])

        adv, ret = compute_gae(
            rew=batch["rew"],
            done=batch["done"],
            v=v,
            v_next=v_next.detach(),
            discount=cfg.discount,
            lam=cfg.gae_lambda,
        )

        # advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = torch.arange(T, device=cfg.device)

        actor_losses = []
        critic_losses = []
        approx_kls = []
        clipfracs = []
        ratio_means = []
        ratio_stds = []
        ent_gate_means = []
        gn_actor_list = []
        gn_critic_list = []

        # gate stats tracking (requested)
        gate_mu_means = []
        gate_mu_stds = []
        gate_logstd_means = []
        gate_logstd_stds = []
        gate_sample_means = []
        gate_sample_stds = []

        for _ep in range(cfg.mini_epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                if mb.numel() == 0:
                    continue

                # old logp
                logp_old = batch["logp_total"][mb]  # (B,)
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                # recompute current logp for each t in minibatch
                logp_new_list = []
                ent_gate_list = []
                v_mb_list = []

                # gate param stats
                mu_list = []
                logstd_list = []
                g_list = []

                for t in mb.tolist():
                    obs_t = obs[t]
                    logits_t, w_logits_t, mu_u_t, logstd_u_t = actor(obs_t)

                    # gate: recompute logprob from stored u (exactly consistent)
                    u_t = batch["u"][t]
                    g_t, logp_gate_per_agent_t, ent_u_per_agent_t = gate_logprob_from_u(
                        mu_u_t, logstd_u_t, u_t
                    )
                    logp_gate_t = logp_gate_per_agent_t.sum()
                    ent_gate_t = ent_u_per_agent_t.mean()

                    # net logprob if netmask
                    if batch["netmask"][t] > 0.5:
                        neigh_t = batch["neighbors"][t]
                        logp_net_t = evaluate_logprob_neighbors_masked(
                            logits=logits_t,
                            neighbors=neigh_t,
                            K=cfg.K,
                            indeg_cap=cfg.indeg_cap,
                        )
                    else:
                        logp_net_t = torch.zeros((), device=cfg.device)

                    logp_total_t = logp_gate_t + logp_net_t

                    logp_new_list.append(logp_total_t)
                    ent_gate_list.append(ent_gate_t)

                    # critic
                    v_mb_list.append(critic(obs_t))

                    # gate stats
                    mu_list.append(mu_u_t.detach())
                    logstd_list.append(logstd_u_t.detach())
                    g_list.append(g_t.detach())

                logp_new = torch.stack(logp_new_list)  # (B,)
                ent_gate = torch.stack(ent_gate_list)  # (B,)
                v_mb = torch.stack(v_mb_list).squeeze(-1)  # (B,)

                ratio = torch.exp(logp_new - logp_old)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Fix (2): gate-only entropy bonus
                ent_bonus = torch.mean(ent_gate)

                actor_loss = policy_loss - cfg.entropy_coef_gate * ent_bonus

                critic_loss = torch.mean((v_mb - ret_mb) ** 2)

                opt_a.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt_a.step()

                opt_c.zero_grad(set_to_none=True)
                (cfg.vf_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                opt_c.step()

                with torch.no_grad():
                    approx_kl = torch.mean(logp_old - logp_new)
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

                # gate stats aggregate over minibatch
                mu_cat = torch.stack(mu_list)        # (B,N)
                ls_cat = torch.stack(logstd_list)    # (B,N)
                g_cat = torch.stack(g_list)          # (B,N)

                gate_mu_means.append(float(mu_cat.mean().item()))
                gate_mu_stds.append(float(mu_cat.std().item()))
                gate_logstd_means.append(float(ls_cat.mean().item()))
                gate_logstd_stds.append(float(ls_cat.std().item()))
                gate_sample_means.append(float(g_cat.mean().item()))
                gate_sample_stds.append(float(g_cat.std().item()))

        # print diagnostics
        if it % 10 == 0:
            dbg = batch["dbg_last"]
            print(
                f"iter={it:04d} | "
                f"actor_loss={np.mean(actor_losses):+.4f} | critic_loss={np.mean(critic_losses):.2f} | "
                f"R_mean={float(batch['rew'].mean().item()):+.4f} | R_std={float(batch['rew'].std().item()):.4f} | "
                f"ratio={np.mean(ratio_means):.3f}±{np.mean(ratio_stds):.3f} | "
                f"kl={np.mean(approx_kls):+.4f} | clipfrac={np.mean(clipfracs):.3f} | "
                f"ent_gate={np.mean(ent_gate_means):.3f} | "
                f"gradA={np.mean(gn_actor_list):.3g} gradC={np.mean(gn_critic_list):.3g} | "
                f"gate_mu={np.mean(gate_mu_means):+.3f}±{np.mean(gate_mu_stds):.3f} "
                f"gate_logstd={np.mean(gate_logstd_means):+.3f}±{np.mean(gate_logstd_stds):.3f} "
                f"gate={np.mean(gate_sample_means):.3f}±{np.mean(gate_sample_stds):.3f} | "
                f"dbg[riskS]={dbg['riskS']:.3g} dbg[flow2]={dbg['flow2']:.3g} dbg[gini]={dbg['gini']:.3g} "
                f"dbg[dP]={dbg['dP']:.3g} dbg[pred]={dbg['pred_err']:.3g}"
            )

            # optional early warning
            if np.mean(approx_kls) > 5 * cfg.target_kl:
                print("WARNING: KL too large (still). Consider lr_actor=2e-5 or mini_epochs=1.")

    print("\nDONE. v10.1 training finished.")
    print("Note: If KL still spikes, reduce lr_actor further or stop updates early when approx_kl exceeds target.\n")


if __name__ == "__main__":
    train_v10_1()
