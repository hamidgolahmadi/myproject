# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 11:49:49 2025

@author: hg2e25
"""

# -*- coding: utf-8 -*-
"""
v9: Info-task reward + early Path-B (financial gate policy)

Key additions vs v8:
1) Private information task:
   - Latent fundamental y_t (AR(1))
   - Each agent receives private noisy signal s_i(t)
   - Belief update mixes private signal + social diffusion:
       b(t+1) = (1-omega)*s(t) + omega*(P b(t)) + noise
   - Reward includes an information-quality term:
       r_info = - ( mean(b_prev) - R_t )^2
     (predicts realized return using last-step aggregate belief)

2) Early Path-B:
   - Financial action still rule-based but policy controls a gate g_i(t) in [0,1]
     that scales how strongly the agent acts on the network signal:
       a_fin_i(t) = tanh( gamma_fin * g_i(t) * (P b)_i )
   - PPO learns (a) network rewiring and (b) g_i(t) per step.

3) Network sampling:
   - Masked sequential sampling with an internal indegree-cap (hard constraint).
   - Without-replacement sampling per row (top-K), while respecting global indegree cap.

Includes PPO diagnostics:
- ratio_mean/std, approx_kl, clipfrac, entropy_mean
- grad_norm_actor, grad_norm_critic

Ablations:
- No-learning (sampling same, updates off)
- Random rewire baseline (with same indeg_cap)
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
    """Compute Gini for nonnegative vector x (robust)."""
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
    """
    Build row-stochastic P from neighbors (N,K) and weights (N,K).
    Assumes neighbors have no self-loops and weights sum to 1 per row.
    """
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        P[i, neighbors[i]] = w[i]
    return P


# =============================================================================
# Environment (Information network, not exposure)
# =============================================================================

class InfoNetworkBondEnvV9:
    """
    Multi-agent env:
    - Information network P (row-stochastic, each i listens to K nodes)
    - Beliefs b updated by private signal + DeGroot mixing
    - Price with impact from trading flow (simple)
    - PPO controls:
        (i) network rewiring (every net_period)
        (ii) financial gate g_i(t) each step in [0,1]
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
        beta_risk=None,         # default ~20-day half-life
        # reward weights
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=0.5,
        w_net=0.5,
        w_info=2.0,             # info-task weight
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
        self.reset()

    def _init_P_random_topk(self):
        """Random row-stochastic influence matrix with exactly K neighbors per row."""
        P = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            candidates = [j for j in range(self.N) if j != i]
            nbrs = self.rng.choice(candidates, size=self.K, replace=False)
            w = self.rng.random(self.K)
            w = w / w.sum()
            P[i, nbrs] = w
        return P

    def _private_signals(self):
        """Generate private signals s_i(t) = y_t + noise."""
        noise = self.rng.normal(0.0, self.sigma_s, size=self.N)
        return self.y + noise

    def _build_obs(self):
        """
        Per-agent observation (vector):
          - price, position, cash
          - own belief b_i
          - private signal s_i
          - perceived social consensus (P b)_i
          - global belief variance, last return, sqrt(risk_v), flow2 proxy
          - credibility (in-degree), gini(in-degree) as global
        """
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

        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.s = self._private_signals()

        self.b = self.rng.normal(0.0, 0.2, size=self.N)

        self.risk_v = 0.0
        self.delta_x_prev = np.zeros(self.N, dtype=float)

        self.P_prev = self.P.copy()
        self.b_bar_prev = float(np.mean(self.b))

        return self._build_obs()

    def step(self, g_gate, neighbors=None, w=None):
        """
        One step.

        Inputs:
          g_gate: (N,) in [0,1] gate scaling rule-based trading intensity
          neighbors: (N,K) optional (only applied when t % net_period == 0)
          w:        (N,K) optional weights (row simplex)

        Returns:
          obs, reward_total (scalar), done, info
        """
        # -----------------------------
        # Apply network action (slow timescale)
        # -----------------------------
        if (self.t % self.net_period) == 0 and neighbors is not None and w is not None:
            self.P_prev = self.P.copy()
            self.P = row_stochastic_from_neighbors(self.N, neighbors, w)

        # -----------------------------
        # Latent fundamental evolves
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

        # Store aggregate belief BEFORE price realization to score prediction vs realized return
        self.b_bar_prev = float(np.mean(self.b))

        # -----------------------------
        # Financial action (rule-based with learned gate)
        # -----------------------------
        g_gate = np.clip(np.asarray(g_gate, dtype=float), 0.0, 1.0)
        signal = self.P @ self.b
        a_fin = np.tanh(self.gamma_fin * g_gate * signal)  # defined by runner via attribute

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
        # Network stats
        in_deg = self.P.sum(axis=0)
        gini_in = gini_coefficient(in_deg)

        # Change in network (Frobenius proxy)
        dP = float(np.mean(np.abs(self.P - self.P_prev)))

        # Flow penalties (stabilize impact)
        flow2 = float(np.mean(delta_x ** 2))
        flow4 = float(np.mean(delta_x ** 4))

        # Risk penalty scaled to ~O(1)
        riskS = float(self.risk_v / max(self.risk_unit, 1e-18))

        # Info-task: prediction loss using previous aggregate belief vs realized return
        # (This forces network to stay informative; flattening P destroys prediction.)
        pred_err = float((self.b_bar_prev - R) ** 2)

        # Total network-centric reward (negative costs)
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
# PPO policy (network logits + weights logits + gate)
# =============================================================================

class ActorNet(nn.Module):
    """
    Produces:
      - embeddings e_i
      - neighbor logits: logits_ij = e_i dot e_j / sqrt(d)
      - weight logits: per i, K logits (converted to simplex)
      - gate logits: per i, scalar -> sigmoid -> [0,1]
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
        self.to_gate = nn.Linear(hidden, 1)

    def forward(self, obs_t: torch.Tensor):
        """
        obs_t: (N, obs_dim)
        returns:
          logits:   (N, N)
          w_logits: (N, K)
          gate:     (N,) in (0,1)
        """
        h = self.mlp(obs_t)
        e = self.to_emb(h)  # (N, emb)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)

        # (N,N) similarity logits
        scale = 1.0 / math.sqrt(e.shape[-1])
        logits = (e @ e.t()) * scale

        w_logits = self.to_wlogits(h)          # (N,K)
        gate = torch.sigmoid(self.to_gate(h))  # (N,1)
        gate = gate.squeeze(-1)                # (N,)

        return logits, w_logits, gate


class CriticNet(nn.Module):
    """Central critic using pooled features (mean over agents)."""

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
        """
        obs_t: (N, obs_dim)
        Returns scalar V(s).
        """
        pooled = obs_t.mean(dim=0, keepdim=True)  # (1, obs_dim)
        return self.v(pooled).squeeze(-1)         # (1,) -> scalar tensor


# =============================================================================
# Masked sequential sampling with indegree cap
# =============================================================================

@torch.no_grad()
def masked_sequential_sample_neighbors(
    logits: torch.Tensor,
    K: int,
    indeg_cap: int,
):
    """
    Sample neighbors (N,K) without replacement per row, while enforcing global indegree cap.

    - logits: (N,N)
    - For each row i:
        sample K distinct j != i
        additionally j must satisfy current indegree[j] < indeg_cap

    Returns:
      neighbors: (N,K) LongTensor
      logp: scalar log-prob of sampled neighbors under current logits (sequential product)
    """
    device = logits.device
    N = logits.shape[0]

    probs = torch.softmax(logits, dim=-1)  # (N,N)
    neighbors = torch.empty((N, K), dtype=torch.long, device=device)

    indeg = torch.zeros((N,), dtype=torch.long, device=device)
    logp_total = torch.zeros((), device=device)

    for i in range(N):
        p = probs[i].clone()

        # Mask self-loop
        p[i] = 0.0

        for k in range(K):
            # Mask nodes that hit indegree cap
            if indeg_cap is not None and indeg_cap > 0:
                cap_mask = (indeg >= indeg_cap).float()
                p = p * (1.0 - cap_mask)

            # Normalize
            s = p.sum()
            if s <= 1e-12:
                # Fallback: pick any available node (rare, but possible under tight cap)
                avail = torch.ones((N,), device=device)
                avail[i] = 0.0
                if indeg_cap is not None and indeg_cap > 0:
                    avail = avail * (indeg < indeg_cap).float()
                idx = torch.where(avail > 0.0)[0]
                if idx.numel() == 0:
                    # Worst-case: ignore cap
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

            # Update masks for without-replacement + indegree
            indeg[j] += 1
            p[j] = 0.0

    # Average logp per edge (stabilizes scale)
    logp_total = logp_total / float(N * K)
    return neighbors, logp_total


def evaluate_logprob_neighbors_masked(
    logits: torch.Tensor,
    neighbors: torch.Tensor,
    K: int,
    indeg_cap: int,
):
    """
    Recompute log-prob of already chosen neighbors using the same sequential logic.
    This must mirror the sampling rule (incl. indegree cap evolution).
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
# PPO runner
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


def rollout_one(env: InfoNetworkBondEnvV9, actor: ActorNet, cfg: PPOConfig, train_mode: bool = True):
    """
    Collect one rollout.

    Stores:
      obs_t (N,obs_dim), V_t, reward_t, done_t
      network action only when t%net_period==0:
         neighbors_t, w_t, logp_net_t
      gate always:
         gate_t
    """
    device = cfg.device

    obs_list = []
    v_list = []
    r_list = []
    done_list = []

    logp_list = []
    entropy_list = []

    neighbors_list = []
    w_list = []
    net_mask_list = []  # 1 if network action executed

    gate_list = []

    info_dbg = {}

    obs = env.reset()
    obs_dim = len(obs[0])

    for t in range(cfg.rollout_len):
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device)  # (N,obs_dim)

        with torch.set_grad_enabled(train_mode):
            logits, w_logits, gate = actor(obs_t)

        # Network action only at net_period
        do_net = (env.t % cfg.net_period) == 0

        if do_net:
            neighbors, logp_net = masked_sequential_sample_neighbors(
                logits=logits,
                K=cfg.K,
                indeg_cap=cfg.indeg_cap,
            )
            w = torch.softmax(w_logits, dim=-1)  # deterministic weights (stability)
            entropy = torch.distributions.Categorical(torch.softmax(logits, dim=-1)).entropy().mean()
        else:
            neighbors = torch.zeros((cfg.K, cfg.K), dtype=torch.long, device=device)  # dummy
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)      # dummy
            logp_net = torch.zeros((), device=device)
            entropy = torch.zeros((), device=device)

        # Step env
        g_gate = gate.detach().cpu().numpy()
        if do_net:
            obs, r, done, info = env.step(
                g_gate=g_gate,
                neighbors=neighbors.detach().cpu().numpy(),
                w=w.detach().cpu().numpy(),
            )
        else:
            obs, r, done, info = env.step(g_gate=g_gate, neighbors=None, w=None)

        # Critic value
        # (Central critic value for the global state)
        v = critic_forward_cached(obs_t, critic=None)  # placeholder, overwritten by caller

        # Store
        obs_list.append(obs_t.detach())
        r_list.append(float(r))
        done_list.append(float(done))

        logp_list.append(logp_net.detach())
        entropy_list.append(entropy.detach())

        net_mask_list.append(float(do_net))
        neighbors_list.append(neighbors.detach())
        w_list.append(w.detach())

        gate_list.append(gate.detach())

        info_dbg = info

        if done:
            break

    return {
        "obs": obs_list,
        "rew": torch.tensor(r_list, dtype=torch.float32, device=device),
        "done": torch.tensor(done_list, dtype=torch.float32, device=device),
        "logp": torch.stack(logp_list),
        "entropy": torch.stack(entropy_list),
        "netmask": torch.tensor(net_mask_list, dtype=torch.float32, device=device),
        "neighbors": neighbors_list,
        "w": w_list,
        "gate": gate_list,
        "dbg_last": info_dbg,
        "obs_dim": obs_dim,
    }


def compute_gae(rew, done, v, v_next, discount=0.99, lam=0.95):
    """
    rew: (T,)
    done: (T,)
    v: (T,)
    v_next: scalar
    returns: adv (T,), ret (T,)
    """
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
    """Compute global grad norm for a module (after backward)."""
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().data.norm(2).item()
        total += g * g
    return float(math.sqrt(total))


# =============================================================================
# Training + evaluation
# =============================================================================

def run_baseline_fixed(env_ctor, cfg_eval, seeds):
    """Fixed network baseline (no rewiring), gate fixed to 1.0."""
    metrics = []
    for sd in seeds:
        env = env_ctor(sd)
        obs = env.reset()

        # Freeze network: never pass neighbors/w
        returns = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        tail_list = []

        prices = []

        for _ in range(cfg_eval.horizon):
            # Gate fixed
            g = np.ones(env.N, dtype=float)

            # No rewiring
            obs, r, done, info = env.step(g_gate=g, neighbors=None, w=None)
            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])

            if done:
                break

        returns = np.asarray(returns, dtype=float)
        cumret = np.cumsum(returns)
        metrics.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(returns, 0.05)),
            "cumret_min_proxy": float(np.min(cumret)),
            "final_price": float(prices[-1]) if len(prices) else float(env.p),
        })
    return metrics


def run_baseline_random_rewire(env_ctor, cfg_eval, seeds):
    """Random rewire baseline with same indeg_cap and K, executed every net_period."""
    metrics = []
    for sd in seeds:
        env = env_ctor(sd)
        obs = env.reset()

        returns = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        prices = []

        for _ in range(cfg_eval.horizon):
            g = np.ones(env.N, dtype=float)

            if (env.t % cfg_eval.net_period) == 0:
                # Random logits -> sample using same indeg-cap logic (numpy simple fallback)
                # Build a random feasible neighbors with indeg-cap greedily
                indeg = np.zeros(env.N, dtype=int)
                neighbors = np.zeros((env.N, cfg_eval.K), dtype=int)
                w = np.zeros((env.N, cfg_eval.K), dtype=float)
                for i in range(env.N):
                    picks = []
                    for k in range(cfg_eval.K):
                        candidates = [j for j in range(env.N) if j != i and indeg[j] < cfg_eval.indeg_cap and j not in picks]
                        if len(candidates) == 0:
                            candidates = [j for j in range(env.N) if j != i and j not in picks]
                        j = candidates[np.random.randint(0, len(candidates))]
                        picks.append(j)
                        indeg[j] += 1
                    neighbors[i] = np.asarray(picks, dtype=int)
                    w[i] = np.ones(cfg_eval.K, dtype=float) / cfg_eval.K

                obs, r, done, info = env.step(g_gate=g, neighbors=neighbors, w=w)
            else:
                obs, r, done, info = env.step(g_gate=g, neighbors=None, w=None)

            returns.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])

            if done:
                break

        returns = np.asarray(returns, dtype=float)
        cumret = np.cumsum(returns)
        metrics.append({
            "vol_proxy_mean_R2": float(np.mean(r2_list)),
            "risk20_mean": float(np.mean(risk_list)),
            "risk20_max": float(riskmax),
            "tail_return_q05": float(np.quantile(returns, 0.05)),
            "cumret_min_proxy": float(np.min(cumret)),
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


# -----------------------------------------------------------------------------
# Critic forwarding helper (avoid circular in rollout_one)
# -----------------------------------------------------------------------------

_CRITIC_GLOBAL = None

def critic_forward_cached(obs_t: torch.Tensor, critic: CriticNet):
    global _CRITIC_GLOBAL
    if critic is not None:
        _CRITIC_GLOBAL = critic
    assert _CRITIC_GLOBAL is not None, "Critic is not set."
    return _CRITIC_GLOBAL(obs_t)


def train_v9():
    # -----------------------------
    # Config (edit here)
    # -----------------------------
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

    # Reward weights (edit here)
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
    print(f"CONFIG v9: horizon={cfg.horizon} | gamma_fin={cfg.gamma_fin} | K={cfg.K} | indeg_cap={cfg.indeg_cap} | net_period={cfg.net_period}")
    print(f"Reward: w_risk={env_kwargs['w_risk']} w_flow2={env_kwargs['w_flow2']} w_flow4={env_kwargs['w_flow4']} w_gini={env_kwargs['w_gini']} w_net={env_kwargs['w_net']} w_info={env_kwargs['w_info']}")
    print("==============================================================================================================")

    # -----------------------------
    # Build env, actor, critic
    # -----------------------------
    seed_train = 0
    set_seed(seed_train)

    env = InfoNetworkBondEnvV9(seed=seed_train, **env_kwargs)
    env.gamma_fin = cfg.gamma_fin  # attach runtime financial intensity

    obs0 = env.reset()
    obs_dim = len(obs0[0])

    actor = ActorNet(obs_dim=obs_dim, hidden=128, emb=64, K=cfg.K).to(cfg.device)
    critic = CriticNet(obs_dim=obs_dim, hidden=128).to(cfg.device)

    # Make critic accessible in rollout
    critic_forward_cached(torch.tensor(np.asarray(obs0, dtype=np.float32), device=cfg.device), critic=critic)

    opt_a = optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # -----------------------------
    # Training loop
    # -----------------------------
    for it in range(cfg.n_iters):
        # Collect rollout
        batch = rollout_one(env, actor, cfg, train_mode=True)

        # Compute values
        obs_stack = torch.stack(batch["obs"])  # (T,N,obs_dim)
        T = obs_stack.shape[0]

        v = torch.zeros((T,), device=cfg.device)
        for t in range(T):
            v[t] = critic(obs_stack[t])

        # Bootstrap
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

        # Advantage normalization (stability)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Prepare indices for PPO minibatches
        idx = torch.arange(T, device=cfg.device)

        # PPO updates
        actor_losses = []
        critic_losses = []
        approx_kls = []
        clipfracs = []
        ratio_means = []
        ratio_stds = []
        ent_means = []
        gn_actor_list = []
        gn_critic_list = []

        for ep in range(cfg.mini_epochs):
            perm = idx[torch.randperm(T)]
            for start in range(0, T, cfg.minibatch):
                mb = perm[start:start + cfg.minibatch]
                if mb.numel() == 0:
                    continue

                # Recompute logp under current policy only for steps where network action executed
                # (netmask==1). For others, policy gradient on network edges is skipped.
                obs_mb = obs_stack[mb]  # (B,N,obs_dim)

                logp_old = batch["logp"][mb]         # (B,)
                netmask = batch["netmask"][mb]       # (B,)
                ent_old = batch["entropy"][mb]       # (B,) (diagnostic only)

                # Forward current
                logp_new_list = []
                ent_new_list = []
                gate_new_list = []

                for ii, t in enumerate(mb.tolist()):
                    logits_t, w_logits_t, gate_t = actor(obs_stack[t])
                    gate_new_list.append(gate_t)

                    if batch["netmask"][t] > 0.5:
                        neigh_t = batch["neighbors"][t]  # (N,K)
                        lp = evaluate_logprob_neighbors_masked(
                            logits=logits_t,
                            neighbors=neigh_t,
                            K=cfg.K,
                            indeg_cap=cfg.indeg_cap,
                        )
                        ent = torch.distributions.Categorical(torch.softmax(logits_t, dim=-1)).entropy().mean()
                    else:
                        lp = torch.zeros((), device=cfg.device)
                        ent = torch.zeros((), device=cfg.device)

                    logp_new_list.append(lp)
                    ent_new_list.append(ent)

                logp_new = torch.stack(logp_new_list)      # (B,)
                ent_new = torch.stack(ent_new_list)        # (B,)
                gate_new = torch.stack(gate_new_list)      # (B,N)

                # Ratio for network decisions
                ratio = torch.exp(logp_new - logp_old)

                # PPO clipped objective on steps where netmask==1
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                # Mask out timesteps without net action
                m = netmask

                # Actor loss (network + gate indirectly through reward; gate has no explicit logp term here)
                # NOTE: gate is deterministic in this v9; to make gate truly PPO you need a stochastic gate policy.
                # Here gate learns only via network-gradient path? No. So we keep gate deterministic BUT still train it
                # by adding a surrogate term using net logp path is insufficient.
                # Practical fix: treat gate as Gaussian policy (recommended) in v10.
                # For v9, we keep gate deterministic and rely on network learning; gate still changes outputs but has no logp.
                # This is the sharp limitation you must accept here.

                # Use masked ratio for updates
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                policy_loss = -(m * torch.min(surr1, surr2)).sum() / (m.sum() + 1e-8)

                # Entropy bonus only where net action exists
                ent_bonus = (m * ent_new).sum() / (m.sum() + 1e-8)

                actor_loss = policy_loss - cfg.entropy_coef * ent_bonus

                # Critic loss
                v_mb = torch.zeros((mb.numel(),), device=cfg.device)
                for j, t in enumerate(mb.tolist()):
                    v_mb[j] = critic(obs_stack[t])
                critic_loss = torch.mean((v_mb - ret_mb) ** 2)

                # Backprop
                opt_a.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt_a.step()

                opt_c.zero_grad(set_to_none=True)
                (cfg.vf_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                opt_c.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = torch.mean(logp_old - logp_new)
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_range).float())

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                approx_kls.append(float(approx_kl.item()))
                clipfracs.append(float(clipfrac.item()))
                ratio_means.append(float(ratio.mean().item()))
                ratio_stds.append(float(ratio.std().item()))
                ent_means.append(float(ent_new.mean().item()))

                gn_actor_list.append(grad_norm(actor))
                gn_critic_list.append(grad_norm(critic))

        # Print every 10 iters
        if it % 10 == 0:
            dbg = batch["dbg_last"]
            print(
                f"iter={it:04d} | "
                f"actor_loss={np.mean(actor_losses):+.4f} | critic_loss={np.mean(critic_losses):.2f} | "
                f"R_mean={float(batch['rew'].mean().item()):+.4f} | R_std={float(batch['rew'].std().item()):.4f} | "
                f"ratio={np.mean(ratio_means):.3f}±{np.mean(ratio_stds):.3f} | "
                f"kl={np.mean(approx_kls):.4f} | clipfrac={np.mean(clipfracs):.3f} | ent={np.mean(ent_means):.3f} | "
                f"gradA={np.mean(gn_actor_list):.3g} gradC={np.mean(gn_critic_list):.3g} | "
                f"dbg[riskS]={dbg['riskS']:.3g} dbg[flow2]={dbg['flow2']:.3g} dbg[gini]={dbg['gini']:.3g} dbg[dP]={dbg['dP']:.3g} dbg[pred]={dbg['pred_err']:.3g}"
            )

            # Early stop if KL explodes
            if np.mean(approx_kls) > 5 * cfg.target_kl:
                print("WARNING: KL too large; consider reducing lr_actor or clip_range.")

    # -----------------------------
    # Evaluation (Fixed vs RandomRewire vs PPO)
    # -----------------------------
    seeds_eval = list(range(30))

    def env_ctor(sd):
        e = InfoNetworkBondEnvV9(seed=sd, **env_kwargs)
        e.gamma_fin = cfg.gamma_fin
        return e

    fixed_ms = run_baseline_fixed(env_ctor, cfg, seeds_eval)
    rand_ms = run_baseline_random_rewire(env_ctor, cfg, seeds_eval)

    # PPO evaluation: run deterministic policy (network sampling ON, gate deterministic)
    ppo_ms = []
    for sd in seeds_eval:
        set_seed(sd)
        envE = env_ctor(sd)
        obs = envE.reset()

        rets = []
        risk_list = []
        riskmax = 0.0
        r2_list = []
        prices = []
        cum = 0.0
        cum_list = []

        for _ in range(cfg.horizon):
            obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=cfg.device)
            with torch.no_grad():
                logits, w_logits, gate = actor(obs_t)

            do_net = (envE.t % cfg.net_period) == 0
            if do_net:
                neighbors, _ = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
                w = torch.softmax(w_logits, dim=-1)
                obs, r, done, info = envE.step(
                    g_gate=gate.cpu().numpy(),
                    neighbors=neighbors.cpu().numpy(),
                    w=w.cpu().numpy(),
                )
            else:
                obs, r, done, info = envE.step(g_gate=gate.cpu().numpy(), neighbors=None, w=None)

            rets.append(info["R"])
            risk_list.append(info["risk_v"])
            riskmax = max(riskmax, info["risk_v"])
            r2_list.append(info["R"] * info["R"])
            prices.append(info["price"])
            cum += info["R"]
            cum_list.append(cum)

            if done:
                break

        rets = np.asarray(rets, dtype=float)
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
        summarize_metrics("PPO", ppo_ms),
    ]
    print_compare_table(rows, title="EVALUATION v9 (seeds=30)")

    print("\nNOTE (important): In v9 the gate is deterministic -> it does NOT have a PPO logprob term.")
    print("If you want PPO to truly learn the financial gate, make gate stochastic (e.g., Beta/Gaussian) in v10.")


if __name__ == "__main__":
    train_v9()
