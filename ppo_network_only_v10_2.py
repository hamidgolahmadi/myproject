# -*- coding: utf-8 -*-
"""
ppo_network_only_v10_2.py  (clean rebuild)

v10.2 = v10.1 + "LOGPROB SCALE: SUM over edges and SUM over agents"
and fixes:
- Gate is truly stochastic (squashed-Gaussian) and its logprob is included in PPO ratio/KL
- Critic output is a true scalar everywhere -> avoids shape/broadcast bugs
- No dead code / placeholders
- PPO diagnostics: ratio mean/std, approx_kl, clipfrac, gate entropy, grad norms, gate stats

Config defaults (edit in train_v10_2()):
- horizon=1000, rollout_len=200, n_iters=200
- K=5, indeg_cap=8, net_period=5
- lr_actor=1e-4, lr_critic=5e-4, clip_range=0.2, mini_epochs=2
- gate logstd clamp in [-2.5, -0.5]
- entropy bonus: gate-only (Normal entropy; not exact after squash, but stable)

NOTE:
- Network rewiring happens only at t % net_period == 0
- Gate action happens every step
- We include both: logp_total = logp_gate_sum_over_agents + logp_net_sum_over_edges
  so ratio/KL should visibly move (not frozen).

@author: Hamid
"""

import math
import numpy as np
from dataclasses import dataclass

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

def row_stochastic_from_neighbors(N: int, neighbors: np.ndarray, w: np.ndarray) -> np.ndarray:
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        P[i, neighbors[i]] = w[i]
    return P


# =============================================================================
# Env (information network + rule-based trading with learned gate)
# =============================================================================

class InfoNetworkBondEnv:
    """
    - Network P: row-stochastic, each i listens to K nodes.
    - Latent fundamental y_t, private signals s_i(t) = y_t + noise.
    - Beliefs b updated via private + social diffusion (DeGroot-ish).
    - Financial action is rule-based BUT scaled by gate g_i(t) in (0,1):
        a_i(t) = tanh( gamma_fin * g_i(t) * (P b)_i )
    - Reward is network-centric (risk/flow/gini/dP) + info-task (prediction error).
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
        beta_risk=None,  # default approx 20 steps memory
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

        # set by trainer
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
                self.p,                # price
                self.x[i],             # position
                self.c[i],             # cash
                self.b[i],             # belief
                self.s[i],             # private signal
                Pb[i],                 # social aggregate (Pb)_i
                in_deg[i],             # indegree
                var_b,                 # global belief variance
                self.R_prev,           # last return
                vol20,                 # sqrt risk state
                mean_abs_deltax,       # mean abs flow
                gini_in,               # global indegree gini
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
        # Apply network action at slow timescale
        if (self.t % self.net_period) == 0 and neighbors is not None and w is not None:
            self.P_prev = self.P.copy()
            self.P = row_stochastic_from_neighbors(self.N, neighbors, w)

        # Fundamental
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y
        self.s = self._private_signals()

        # Beliefs
        Pb_old = self.P @ self.b
        eta_b = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.omega_social) * self.s + self.omega_social * Pb_old + eta_b

        # store aggregate belief before price shock
        self.b_bar_prev = float(np.mean(self.b))

        # Financial action (rule-based * gate)
        g_gate = np.clip(np.asarray(g_gate, dtype=float), 0.0, 1.0)
        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * g_gate * signal)

        delta_x = a_fin * self.x_max
        tc = self.tau * np.abs(delta_x)

        self.x = self.x + delta_x

        # Price
        net_flow = float(np.sum(delta_x))
        eps_p = float(self.rng.normal(0.0, self.sigma_eps))

        self.p_prev = self.p
        self.p = self.p + self.kappa * net_flow + eps_p
        self.p = max(self.p, 1e-6)

        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        # Risk (EWMA of R^2)
        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        # Cash update
        self.c = self.c - self.p * delta_x - tc
        self.delta_x_prev = delta_x.copy()

        # Reward parts
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

class ActorNet(nn.Module):
    """
    Outputs:
      - network logits (N,N) from cosine similarity in embedding space
      - weight logits (N,K) (row simplex via softmax)
      - gate distribution params per agent: mu, logstd (pre-sigmoid)
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
        """
        obs_t: (N, obs_dim)
        returns:
          logits:   (N, N)
          w_logits: (N, K)
          gate_mu:  (N,)
          gate_logstd: (N,)
        """
        h = self.mlp(obs_t)
        e = self.to_emb(h)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        logits = (e @ e.t()) * (1.0 / math.sqrt(e.shape[-1]))

        w_logits = self.to_wlogits(h)

        gate_mu = self.to_gate_mu(h).squeeze(-1)         # (N,)
        gate_logstd = self.to_gate_logstd(h).squeeze(-1) # (N,)

        return logits, w_logits, gate_mu, gate_logstd


class CriticNet(nn.Module):
    """Central critic: pool by mean over agents -> scalar value."""
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
        pooled = obs_t.mean(dim=0, keepdim=True)  # (1, obs_dim)
        out = self.v(pooled)                      # (1,1)
        return out.squeeze()                      # 0-dim scalar


# =============================================================================
# Network sampling with indegree cap (sequential masking)
# =============================================================================

@torch.no_grad()
def masked_sequential_sample_neighbors(logits: torch.Tensor, K: int, indeg_cap: int):
    """
    Sample neighbors (N,K) without replacement per row, enforcing global indegree cap.
    Returns:
      neighbors: (N,K) LongTensor
      logp_edges: scalar = SUM over selected edges of log prob (NOT averaged)
    """
    device = logits.device
    N = logits.shape[0]
    probs = torch.softmax(logits, dim=-1)

    neighbors = torch.empty((N, K), dtype=torch.long, device=device)
    indeg = torch.zeros((N,), dtype=torch.long, device=device)

    logp_total = torch.zeros((), device=device)

    for i in range(N):
        p = probs[i].clone()
        p[i] = 0.0  # forbid self

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

    return neighbors, logp_total


def evaluate_logprob_neighbors_masked(logits: torch.Tensor, neighbors: torch.Tensor, K: int, indeg_cap: int):
    """
    Recompute logprob of chosen neighbors under current logits,
    mirroring the same sequential indegree-cap logic.
    Returns logp_edges SUM over edges (NOT averaged).
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

    return logp_total


# =============================================================================
# Gate: squashed Gaussian logprob (sigmoid)
# =============================================================================

def logprob_squashed_sigmoid_gaussian(u: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor, eps: float = 1e-8):
    """
    u is the pre-sigmoid action (same shape as mu/logstd), sampled from N(mu, std).
    g = sigmoid(u)

    log p(g) = log p(u) - log|dg/du|
             = log N(u; mu, std) - log( sigmoid(u)*(1-sigmoid(u)) )
    Returns:
      logp: same shape as u
      g:    sigmoid(u)
    """
    std = torch.exp(logstd)
    dist = torch.distributions.Normal(mu, std)
    logp_u = dist.log_prob(u)

    g = torch.sigmoid(u)
    jac = g * (1.0 - g)
    logp = logp_u - torch.log(jac + eps)
    return logp, g

def normal_entropy(mu: torch.Tensor, logstd: torch.Tensor):
    """
    Entropy of Normal(mu, std): 0.5*log(2*pi*e*std^2)
    Returns per-dimension entropy (same shape as mu).
    """
    return 0.5 * (1.0 + math.log(2.0 * math.pi)) + logstd


# =============================================================================
# PPO
# =============================================================================

@dataclass
class PPOConfig:
    horizon: int = 1000
    rollout_len: int = 200
    n_iters: int = 200

    gamma_fin: float = 5.0
    net_period: int = 5
    K: int = 5
    indeg_cap: int = 8

    mini_epochs: int = 2
    minibatch: int = 64

    lr_actor: float = 1e-4
    lr_critic: float = 5e-4

    clip_range: float = 0.2
    gae_lambda: float = 0.95
    discount: float = 0.99

    entropy_coef_gate: float = 0.01  # gate-only entropy bonus
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0

    # gate logstd clamp
    gate_logstd_min: float = -2.5
    gate_logstd_max: float = -0.5

    device: str = "cpu"


def compute_gae(rew, done, v, v_last, discount=0.99, lam=0.95):
    """
    rew: (T,)
    done: (T,)
    v: (T,)
    v_last: scalar (bootstrap)
    returns adv (T,), ret (T,)
    """
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    gae = 0.0
    v_next = v_last
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


def rollout_one(env: InfoNetworkBondEnv, actor: ActorNet, critic: CriticNet, cfg: PPOConfig):
    """
    Collect one rollout of length cfg.rollout_len (or until done).

    Stores per t:
      obs_t (N,obs_dim)
      v_t (scalar)
      r_t (scalar)
      done_t (scalar)
      netmask_t (0/1)
      neighbors_t (N,K) when netmask=1 else dummy
      w_t (N,K) when netmask=1 else dummy
      logp_net_t (scalar; sum over edges) -> 0 if netmask=0
      gate params mu/logstd (N,)
      gate action u (N,) and g (N,)
      logp_gate_t (scalar; sum over agents)
      ent_gate_t (scalar; sum over agents, Normal entropy)
      logp_total_t = logp_net_t + logp_gate_t
    """
    device = cfg.device
    obs = env.reset()
    obs_dim = len(obs[0])

    obs_list = []
    v_list = []
    r_list = []
    done_list = []

    netmask_list = []
    neighbors_list = []
    w_list = []
    logp_net_list = []

    gate_mu_list = []
    gate_logstd_list = []
    gate_u_list = []
    gate_g_list = []
    logp_gate_list = []
    ent_gate_list = []

    logp_total_list = []

    dbg_last = None

    for _ in range(cfg.rollout_len):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)  # (N,obs_dim)

        with torch.no_grad():
            logits, w_logits, mu, logstd_raw = actor(obs_t)

            # clamp logstd
            logstd = torch.clamp(logstd_raw, cfg.gate_logstd_min, cfg.gate_logstd_max)

            # sample u ~ N(mu,std)
            std = torch.exp(logstd)
            u = mu + std * torch.randn_like(mu)

            # logprob + squashed gate
            logp_gate_vec, g = logprob_squashed_sigmoid_gaussian(u, mu, logstd)
            logp_gate = logp_gate_vec.sum()  # SUM over agents

            # entropy (normal) sum over agents (gate-only bonus)
            ent_gate = normal_entropy(mu, logstd).sum()

            # network action (slow)
            do_net = (env.t % cfg.net_period) == 0
            if do_net:
                neighbors, logp_net = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
                w = torch.softmax(w_logits, dim=-1)  # deterministic weights
            else:
                neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
                w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)
                logp_net = torch.zeros((), device=device)

            # step env
            if do_net:
                obs, r, done, info = env.step(
                    g_gate=g.detach().cpu().numpy(),
                    neighbors=neighbors.detach().cpu().numpy(),
                    w=w.detach().cpu().numpy(),
                )
            else:
                obs, r, done, info = env.step(g_gate=g.detach().cpu().numpy(), neighbors=None, w=None)

            v = critic(obs_t).detach().squeeze()  # scalar

            logp_total = (logp_gate + logp_net).detach()

        # store
        obs_list.append(obs_t.detach())
        v_list.append(v)

        r_list.append(float(r))
        done_list.append(float(done))

        netmask_list.append(float(do_net))
        neighbors_list.append(neighbors.detach())
        w_list.append(w.detach())
        logp_net_list.append(logp_net.detach())

        gate_mu_list.append(mu.detach())
        gate_logstd_list.append(logstd.detach())
        gate_u_list.append(u.detach())
        gate_g_list.append(g.detach())
        logp_gate_list.append(logp_gate.detach())
        ent_gate_list.append(ent_gate.detach())

        logp_total_list.append(logp_total)

        dbg_last = info

        if done:
            break

    T = len(r_list)
    batch = {
        "obs": torch.stack(obs_list, dim=0),                 # (T,N,obs_dim)
        "v": torch.stack(v_list, dim=0).view(-1),           # (T,)
        "rew": torch.tensor(r_list, dtype=torch.float32, device=device).view(-1),   # (T,)
        "done": torch.tensor(done_list, dtype=torch.float32, device=device).view(-1),# (T,)

        "netmask": torch.tensor(netmask_list, dtype=torch.float32, device=device).view(-1),  # (T,)
        "neighbors": neighbors_list,  # list length T, each (N,K)
        "w": w_list,                  # list length T, each (N,K)
        "logp_net": torch.stack(logp_net_list, dim=0).view(-1),  # (T,)

        "gate_u": gate_u_list,         # list length T, each (N,)
        "gate_g": gate_g_list,         # list length T, each (N,)
        "logp_gate": torch.stack(logp_gate_list, dim=0).view(-1),# (T,)
        "ent_gate": torch.stack(ent_gate_list, dim=0).view(-1),  # (T,)

        "logp_total": torch.stack(logp_total_list, dim=0).view(-1),# (T,)
        "dbg_last": dbg_last,
        "obs_dim": obs_dim,
    }
    return batch


# =============================================================================
# Baselines + Evaluation
# =============================================================================

def run_baseline_fixed(env_ctor, cfg: PPOConfig, seeds):
    metrics = []
    for sd in seeds:
        env = env_ctor(sd)
        obs = env.reset()

        returns = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        prices = []
        cum = 0.0
        cum_list = []

        for _ in range(cfg.horizon):
            g = np.ones(env.N, dtype=float)
            obs, r, done, info = env.step(g_gate=g, neighbors=None, w=None)
            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])
            cum += info["R"]
            cum_list.append(cum)
            if done:
                break

        returns = np.asarray(returns, dtype=float)
        metrics.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(returns, 0.05)),
            "cumret_min_proxy": float(np.min(np.asarray(cum_list, dtype=float))),
            "final_price": float(prices[-1]) if len(prices) else float(env.p),
        })
    return metrics


def run_baseline_random_rewire(env_ctor, cfg: PPOConfig, seeds):
    metrics = []
    for sd in seeds:
        env = env_ctor(sd)
        obs = env.reset()

        returns = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        prices = []
        cum = 0.0
        cum_list = []

        for _ in range(cfg.horizon):
            g = np.ones(env.N, dtype=float)

            if (env.t % cfg.net_period) == 0:
                indeg = np.zeros(env.N, dtype=int)
                neighbors = np.zeros((env.N, cfg.K), dtype=int)
                w = np.zeros((env.N, cfg.K), dtype=float)

                for i in range(env.N):
                    picks = []
                    for _k in range(cfg.K):
                        candidates = [j for j in range(env.N) if j != i and j not in picks and indeg[j] < cfg.indeg_cap]
                        if len(candidates) == 0:
                            candidates = [j for j in range(env.N) if j != i and j not in picks]
                        j = candidates[np.random.randint(0, len(candidates))]
                        picks.append(j)
                        indeg[j] += 1
                    neighbors[i] = np.asarray(picks, dtype=int)
                    w[i] = np.ones(cfg.K, dtype=float) / cfg.K

                obs, r, done, info = env.step(g_gate=g, neighbors=neighbors, w=w)
            else:
                obs, r, done, info = env.step(g_gate=g, neighbors=None, w=None)

            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])
            cum += info["R"]
            cum_list.append(cum)
            if done:
                break

        returns = np.asarray(returns, dtype=float)
        metrics.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(returns, 0.05)),
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
# Train v10.2
# =============================================================================

def train_v10_2():
    cfg = PPOConfig(
        horizon=1000,
        rollout_len=200,
        n_iters=200,
        gamma_fin=5.0,
        net_period=5,
        K=5,
        indeg_cap=8,

        lr_actor=1e-4,
        lr_critic=5e-4,
        clip_range=0.2,
        mini_epochs=2,
        minibatch=64,

        entropy_coef_gate=0.01,
        gate_logstd_min=-2.5,
        gate_logstd_max=-0.5,
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
    print(f"CONFIG v10.2: horizon={cfg.horizon} | gamma_fin={cfg.gamma_fin} | K={cfg.K} | indeg_cap={cfg.indeg_cap} | net_period={cfg.net_period}")
    print(f"Reward: w_risk={env_kwargs['w_risk']} w_flow2={env_kwargs['w_flow2']} w_flow4={env_kwargs['w_flow4']} w_gini={env_kwargs['w_gini']} w_net={env_kwargs['w_net']} w_info={env_kwargs['w_info']}")
    print(f"PPO: lr_actor={cfg.lr_actor} clip_range={cfg.clip_range} mini_epochs={cfg.mini_epochs} | Gate logstd in [{cfg.gate_logstd_min}, {cfg.gate_logstd_max}] | Entropy bonus: gate-only")
    print("LOGPROB SCALE: SUM over edges and SUM over agents (ratio/KL should move).")
    print("==============================================================================================================")

    seed_train = 0
    set_seed(seed_train)

    env = InfoNetworkBondEnv(seed=seed_train, **env_kwargs)
    env.gamma_fin = cfg.gamma_fin

    obs0 = env.reset()
    obs_dim = len(obs0[0])

    actor = ActorNet(obs_dim=obs_dim, hidden=128, emb=64, K=cfg.K).to(cfg.device)
    critic = CriticNet(obs_dim=obs_dim, hidden=128).to(cfg.device)

    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    for it in range(cfg.n_iters):
        batch = rollout_one(env, actor, critic, cfg)
        T = batch["rew"].shape[0]

        # values (already stored in rollout as v, but recompute once for safety/stability)
        with torch.no_grad():
            v = batch["v"].view(-1)  # (T,)
            v_last = critic(batch["obs"][-1]).detach().squeeze() * (1.0 - batch["done"][-1])

        adv, ret = compute_gae(
            rew=batch["rew"],
            done=batch["done"],
            v=v.detach(),
            v_last=v_last.detach(),
            discount=cfg.discount,
            lam=cfg.gae_lambda,
        )
        adv = adv.view(-1)
        ret = ret.view(-1)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # old logp totals
        logp_old = batch["logp_total"].detach().view(-1)

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

        gate_mu_stats = []
        gate_logstd_stats = []
        gate_g_stats = []

        for _ep in range(cfg.mini_epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                if mb.numel() == 0:
                    continue

                obs_mb = batch["obs"][mb]     # (B,N,obs_dim)
                adv_mb = adv[mb]              # (B,)
                ret_mb = ret[mb].view(-1)     # (B,)
                logp_old_mb = logp_old[mb]    # (B,)

                logp_new_list = []
                ent_gate_list = []
                mu_list = []
                logstd_list = []
                g_list = []

                # recompute current logp for same actions stored in batch:
                # - gate action stored as u_t (pre-sigmoid)
                # - neighbors stored for network steps
                for ii, t in enumerate(mb.tolist()):
                    obs_t = obs_mb[ii]
                    logits_t, w_logits_t, mu_t, logstd_raw_t = actor(obs_t)
                    logstd_t = torch.clamp(logstd_raw_t, cfg.gate_logstd_min, cfg.gate_logstd_max)

                    # gate: reuse stored u (action) to compute logprob under current params
                    u_old = batch["gate_u"][t].to(cfg.device)  # (N,)
                    logp_gate_vec_new, g_new = logprob_squashed_sigmoid_gaussian(u_old, mu_t, logstd_t)
                    logp_gate_new = logp_gate_vec_new.sum()  # SUM over agents

                    ent_gate_new = normal_entropy(mu_t, logstd_t).sum()

                    # network: only if net action executed at that timestep
                    if batch["netmask"][t].item() > 0.5:
                        neigh_t = batch["neighbors"][t].to(cfg.device)  # (N,K)
                        logp_net_new = evaluate_logprob_neighbors_masked(
                            logits=logits_t,
                            neighbors=neigh_t,
                            K=cfg.K,
                            indeg_cap=cfg.indeg_cap,
                        )
                    else:
                        logp_net_new = torch.zeros((), device=cfg.device)

                    logp_total_new = logp_gate_new + logp_net_new

                    logp_new_list.append(logp_total_new)
                    ent_gate_list.append(ent_gate_new)

                    mu_list.append(mu_t.detach())
                    logstd_list.append(logstd_t.detach())
                    g_list.append(g_new.detach())

                logp_new = torch.stack(logp_new_list).view(-1)  # (B,)
                ent_gate_mb = torch.stack(ent_gate_list).view(-1)  # (B,)

                ratio = torch.exp(logp_new - logp_old_mb)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # entropy bonus gate-only
                ent_bonus = torch.mean(ent_gate_mb)
                actor_loss = policy_loss - cfg.entropy_coef_gate * ent_bonus

                # critic loss
                v_mb = torch.zeros((mb.numel(),), device=cfg.device)
                for j in range(mb.numel()):
                    v_mb[j] = critic(obs_mb[j]).squeeze()
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

                # gate stats (diagnostic only)
                mu_cat = torch.stack(mu_list, dim=0)         # (B,N)
                ls_cat = torch.stack(logstd_list, dim=0)     # (B,N)
                g_cat = torch.stack(g_list, dim=0)           # (B,N)

                gate_mu_stats.append((float(mu_cat.mean().item()), float(mu_cat.std().item())))
                gate_logstd_stats.append((float(ls_cat.mean().item()), float(ls_cat.std().item())))
                gate_g_stats.append((float(g_cat.mean().item()), float(g_cat.std().item())))

        # print every 10 iters
        if it % 10 == 0:
            dbg = batch["dbg_last"]
            mu_m, mu_s = gate_mu_stats[-1]
            ls_m, ls_s = gate_logstd_stats[-1]
            g_m, g_s = gate_g_stats[-1]
            print(
                f"iter={it:04d} | "
                f"actor_loss={np.mean(actor_losses):+.4f} | critic_loss={np.mean(critic_losses):.2f} | "
                f"R_mean={float(batch['rew'].mean().item()):+.4f} | R_std={float(batch['rew'].std().item()):.4f} | "
                f"ratio={np.mean(ratio_means):.3f}±{np.mean(ratio_stds):.3f} | "
                f"kl={np.mean(approx_kls):+.4f} | clipfrac={np.mean(clipfracs):.3f} | "
                f"ent_gate={np.mean(ent_gate_means):.3f} | "
                f"gradA={np.mean(gn_actor_list):.3g} gradC={np.mean(gn_critic_list):.3g} | "
                f"gate_mu={mu_m:+.3f}±{mu_s:.3f} gate_logstd={ls_m:+.3f}±{ls_s:.3f} gate={g_m:.3f}±{g_s:.3f} | "
                f"dbg[riskS]={dbg['riskS']:.3g} dbg[flow2]={dbg['flow2']:.3g} dbg[gini]={dbg['gini']:.3g} "
                f"dbg[dP]={dbg['dP']:.3g} dbg[pred]={dbg['pred_err']:.3g}"
            )

    # -----------------------------
    # Evaluation
    # -----------------------------
    seeds_eval = list(range(30))

    def env_ctor(sd):
        e = InfoNetworkBondEnv(seed=sd, **env_kwargs)
        e.gamma_fin = cfg.gamma_fin
        return e

    fixed_ms = run_baseline_fixed(env_ctor, cfg, seeds_eval)
    rand_ms = run_baseline_random_rewire(env_ctor, cfg, seeds_eval)

    # PPO eval: sample network at net steps, sample gate from learned distribution
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
                logits, w_logits, mu, logstd_raw = actor(obs_t)
                logstd = torch.clamp(logstd_raw, cfg.gate_logstd_min, cfg.gate_logstd_max)
                std = torch.exp(logstd)
                u = mu + std * torch.randn_like(mu)
                _, g = logprob_squashed_sigmoid_gaussian(u, mu, logstd)

                do_net = (envE.t % cfg.net_period) == 0
                if do_net:
                    neighbors, _ = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
                    w = torch.softmax(w_logits, dim=-1)
                    obs, r, done, info = envE.step(
                        g_gate=g.cpu().numpy(),
                        neighbors=neighbors.cpu().numpy(),
                        w=w.cpu().numpy(),
                    )
                else:
                    obs, r, done, info = envE.step(g_gate=g.cpu().numpy(), neighbors=None, w=None)

            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])
            cum += info["R"]
            cum_list.append(cum)
            if done:
                break

        returns = np.asarray(returns, dtype=float)
        ppo_ms.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(returns, 0.05)),
            "cumret_min_proxy": float(np.min(np.asarray(cum_list, dtype=float))),
            "final_price": float(prices[-1]) if len(prices) else float(envE.p),
        })

    rows = [
        summarize_metrics("Fixed", fixed_ms),
        summarize_metrics("RandomRewire", rand_ms),
        summarize_metrics("PPO", ppo_ms),
    ]
    print_compare_table(rows, title="EVALUATION v10.2 (seeds=30)")


if __name__ == "__main__":
    train_v10_2()
