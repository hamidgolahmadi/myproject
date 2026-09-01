# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 13:59:10 2025

@author: hg2e25
"""

# -*- coding: utf-8 -*-
"""
PPO (Network-only) v2 for InfoNetworkBondEnv

Path A:
- Financial action is fixed (rule-based)
- PPO learns ONLY network rewiring every net_period days (SMDP macro-step)
- Includes:
  * reward scaling
  * correct GAE (uses V_{t+1})
  * evaluation vs baselines (Fixed, Similarity-Rewire, PPO-Learned)

Requirements:
- numpy
- torch

Usage:
- Make sure InfoNetworkBondEnv is importable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 0) IMPORT ENV (adjust this import to your setup)
# ============================================================
# Option A: If InfoNetworkBondEnv is in the same file, comment this import.
# Option B: If env is in temp.py, use:
# from temp import InfoNetworkBondEnv

from temp import InfoNetworkBondEnv  # <-- change if needed


# ============================================================
# 1) FIXED FINANCIAL POLICY (constant)
# ============================================================

def fixed_financial_policy(env, gamma=5.0):
    """Belief-driven trading, fixed across all experiments."""
    neigh_signal = env.P @ env.b
    return np.tanh(gamma * neigh_signal)


# ============================================================
# 2) NETWORK STATE (compact, per-agent)
# ============================================================

def build_net_state(env):
    """
    Per-agent state at macro decision times.
    Shape: (N, d=6)

    Features:
    - b_i
    - m_i = (P b)_i
    - in_influence_i = sum_k P[k,i]
    - global risk_v
    - global var(b)
    - last return R_prev
    """
    b = env.b
    m = env.P @ env.b
    in_infl = env.P.sum(axis=0)

    risk_v = float(env.risk_v)
    var_b = float(np.var(env.b))
    R_prev = float(env.R_prev)

    N = env.N
    x = np.zeros((N, 6), dtype=np.float32)
    x[:, 0] = b
    x[:, 1] = m
    x[:, 2] = in_infl
    x[:, 3] = risk_v
    x[:, 4] = var_b
    x[:, 5] = R_prev
    return x


# ============================================================
# 3) ACTOR + CRITIC
# ============================================================

class NetActor(nn.Module):
    """
    Shared actor (parameter sharing).
    Input: (N, d)
    Output:
      - neighbor logits per agent: (N, N)
      - weight logits per agent:   (N, K)
    """

    def __init__(self, d_in, N, K, hidden=128):
        super().__init__()
        self.N = N
        self.K = K

        self.backbone = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.head_logits = nn.Linear(hidden, N)
        self.head_w = nn.Linear(hidden, K)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.head_logits(h)    # (N,N)
        w_logits = self.head_w(h)       # (N,K)
        return logits, w_logits


class NetCritic(nn.Module):
    """
    Centralized critic:
    Aggregates per-agent states by mean -> V(s).
    """

    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_agents):
        xg = x_agents.mean(dim=0)       # (d,)
        return self.v(xg).squeeze(-1)   # scalar


# ============================================================
# 4) ACTION: SAMPLE TOP-K NEIGHBORS + WEIGHTS
# ============================================================

def sample_topk_without_replacement(probs_row, K):
    """
    Sample K indices without replacement from a probability vector.
    Simple loop (fine for N~50).
    Returns:
      idx_list: list[int] length K
      logp_sum: torch scalar
    """
    p = probs_row.clone()
    idx_list = []
    logp_sum = torch.zeros((), device=p.device)

    for _ in range(K):
        idx = torch.multinomial(p, num_samples=1).item()
        idx_list.append(idx)
        logp_sum = logp_sum + torch.log(p[idx] + 1e-12)

        p[idx] = 0.0
        p = p / (p.sum() + 1e-12)

    return idx_list, logp_sum


def sample_network_action(logits, w_logits, K):
    """
    Sample network action for all agents.

    Returns:
      neighbors: (N,K) LongTensor
      w:         (N,K) Tensor simplex weights
      logp:      scalar log-prob for neighbors (sum over i,k), used in PPO
    """
    N = logits.shape[0]
    device = logits.device

    probs = torch.softmax(logits, dim=-1)  # (N,N)

    neighbors = torch.zeros((N, K), dtype=torch.long, device=device)
    logp_total = torch.zeros((), device=device)

    for i in range(N):
        p = probs[i].clone()

        # Forbid self-loop
        p[i] = 0.0
        p = p / (p.sum() + 1e-12)

        idx_list, logp_i = sample_topk_without_replacement(p, K)
        neighbors[i] = torch.tensor(idx_list, dtype=torch.long, device=device)
        logp_total = logp_total + logp_i

    # Deterministic weights (v2): keep simple and stable
    w = torch.softmax(w_logits, dim=-1)  # (N,K)

    return neighbors, w, logp_total


def evaluate_logprob_neighbors(logits, neighbors):
    """
    Compute log-prob of the SAME chosen neighbors under current logits.
    Uses the same without-replacement approximation as sampling.
    Returns scalar total logp.
    """
    N, K = neighbors.shape
    probs = torch.softmax(logits, dim=-1)

    logp_total = torch.zeros((), device=logits.device)

    for i in range(N):
        p = probs[i].clone()
        p[i] = 0.0
        p = p / (p.sum() + 1e-12)

        chosen = neighbors[i].tolist()
        for j in chosen:
            logp_total = logp_total + torch.log(p[j] + 1e-12)
            p[j] = 0.0
            p = p / (p.sum() + 1e-12)

    return logp_total


def build_env_net_inputs(env, neighbors, w):
    """
    Convert sampled (neighbors, w) into env inputs:
      net_logits: (N,N) such that env argsort picks chosen neighbors
      net_w_logits: (N,K) logits for weights (env softmax -> weights)
    """
    N = env.N
    K = env.K

    net_logits = -1e9 * np.ones((N, N), dtype=np.float32)
    for i in range(N):
        # Forbid self
        net_logits[i, i] = -np.inf
        for k in range(K):
            j = int(neighbors[i, k].item())
            net_logits[i, j] = 0.0

    w_np = w.detach().cpu().numpy().astype(np.float32)
    net_w_logits = np.log(w_np + 1e-12)  # consistent with softmax

    return net_logits, net_w_logits


# ============================================================
# 5) MACRO-STEP (SMDP): ONE NETWORK ACTION -> net_period MICRO STEPS
# ============================================================

@torch.no_grad()
def macro_step(env, actor, critic, gamma_fin=5.0, reward_scale=1e6, device="cpu"):
    """
    One macro-step:
    - choose network rewiring once
    - simulate env for net_period days with fixed financial policy

    Macro reward (network objective):
      R = - sum_{micro} risk_v(micro)   (scaled)

    Returns dict with:
      s, neighbors, logp, R, done, V
    """
    # State at decision time
    s_np = build_net_state(env)  # (N,d)
    s = torch.tensor(s_np, device=device)

    # Actor outputs + sample action
    logits, w_logits = actor(s)
    neighbors, w, logp = sample_network_action(logits, w_logits, env.K)

    # Value
    V = critic(s)

    # Prepare env inputs
    net_logits_np, net_w_logits_np = build_env_net_inputs(env, neighbors, w)

    # Roll micro-steps
    R_macro = 0.0
    done = False

    for _ in range(env.net_period):
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)

        if (env.t % env.net_period) == 0:
            _, _, done, info = env.step(a_fin, net_logits_np, net_w_logits_np)
        else:
            _, _, done, info = env.step(a_fin, None, None)

        # Risk penalty accumulates (shared market outcome)
        R_macro += -float(info["risk_v"])

        if done:
            break

    # Scale reward for stable learning
    R_macro = reward_scale * R_macro

    return {
        "s": s,
        "neighbors": neighbors.detach(),
        "logp": logp.detach(),
        "R": torch.tensor(R_macro, dtype=torch.float32, device=device),
        "done": torch.tensor(float(done), dtype=torch.float32, device=device),
        "V": V.detach(),
    }


# ============================================================
# 6) ROLLOUT BUFFER + GAE (correct)
# ============================================================

class Rollout:
    """Stores macro-step transitions for PPO."""
    def __init__(self):
        self.s = []
        self.neighbors = []
        self.logp = []
        self.R = []
        self.done = []
        self.V = []

    def add(self, tr):
        self.s.append(tr["s"])
        self.neighbors.append(tr["neighbors"])
        self.logp.append(tr["logp"])
        self.R.append(tr["R"])
        self.done.append(tr["done"])
        self.V.append(tr["V"])

    def stack(self, device="cpu"):
        logp = torch.stack(self.logp).to(device)   # (T,)
        R = torch.stack(self.R).to(device)         # (T,)
        done = torch.stack(self.done).to(device)   # (T,)
        V = torch.stack(self.V).to(device)         # (T,)
        return logp, R, done, V


def compute_returns_advantages(R, done, V, gamma=0.99, lam=0.95):
    """
    Proper GAE(λ) on macro-steps using V_{t+1}.
    """
    T = R.shape[0]
    adv = torch.zeros_like(R)

    V_next = torch.zeros_like(V)
    V_next[:-1] = V[1:]
    V_next[-1] = 0.0

    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = R[t] + gamma * V_next[t] * mask - V[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae

    ret = adv + V
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return ret, adv


# ============================================================
# 7) PPO UPDATE (correct logprob of stored actions)
# ============================================================

def ppo_update(actor, critic, optA, optC, rollout, device="cpu",
               clip=0.2, vf_coef=0.5, ent_coef=0.0,
               gamma=0.99, lam=0.95, epochs=4):
    """
    PPO update on macro-steps.
    Only neighbor-selection logprob is used (weights deterministic).
    """
    logp_old, R, done, V_old = rollout.stack(device=device)
    ret, adv = compute_returns_advantages(R, done, V_old, gamma=gamma, lam=lam)

    T = len(rollout.s)

    for _ in range(epochs):
        actor_loss = 0.0
        critic_loss = 0.0

        for t in range(T):
            s = rollout.s[t].to(device)                         # (N,d)
            neighbors = rollout.neighbors[t].to(device)         # (N,K)

            logits, w_logits = actor(s)
            logp_new = evaluate_logprob_neighbors(logits, neighbors)

            ratio = torch.exp(logp_new - logp_old[t])
            surr1 = ratio * adv[t]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[t]
            actor_loss = actor_loss + (-torch.min(surr1, surr2))

            V = critic(s)
            critic_loss = critic_loss + (ret[t] - V).pow(2)

        actor_loss = actor_loss / T
        critic_loss = critic_loss / T

        # Update actor
        optA.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optA.step()

        # Update critic
        optC.zero_grad()
        (vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optC.step()

    return {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "R_mean": float(R.detach().cpu().mean().item()),
        "R_std": float(R.detach().cpu().std(unbiased=False).item()),
        "adv_mean": float(adv.detach().cpu().mean().item()),
        "adv_std": float(adv.detach().cpu().std(unbiased=False).item()),
    }


# ============================================================
# 8) TRAINING LOOP
# ============================================================

def train_network_only_ppo(
    make_env,
    iters=200,
    macro_steps_per_iter=64,
    gamma_fin=5.0,
    reward_scale=1e6,
    lr=3e-4,
    device="cpu",
):
    """
    Trains network-only PPO with macro-step rollouts.
    """
    env0 = make_env(seed=0)
    d_in = build_net_state(env0).shape[1]

    actor = NetActor(d_in=d_in, N=env0.N, K=env0.K, hidden=128).to(device)
    critic = NetCritic(d_in=d_in, hidden=128).to(device)

    optA = optim.Adam(actor.parameters(), lr=lr)
    optC = optim.Adam(critic.parameters(), lr=lr)

    for it in range(iters):
        env = make_env(seed=int(np.random.randint(0, 10_000)))
        rollout = Rollout()

        for _ in range(macro_steps_per_iter):
            if env.t >= env.horizon:
                env.reset()

            tr = macro_step(
                env, actor, critic,
                gamma_fin=gamma_fin,
                reward_scale=reward_scale,
                device=device
            )
            rollout.add(tr)

        stats = ppo_update(
            actor, critic, optA, optC, rollout,
            device=device,
            clip=0.2, vf_coef=0.5, ent_coef=0.0,
            gamma=0.99, lam=0.95, epochs=4
        )

        if (it % 10) == 0:
            print(
                f"iter={it:04d} | "
                f"actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.6f} | "
                f"R_mean={stats['R_mean']:.4g} | R_std={stats['R_std']:.4g} | "
                f"adv_std={stats['adv_std']:.4g}"
            )

    return actor, critic


# ============================================================
# 9) EVALUATION (baselines vs learned)
# ============================================================

def gini(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    if x.sum() < eps:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return 1.0 + 1.0 / n - 2.0 * np.sum(cum) / (n * cum[-1] + eps)


def coordination_metric(P, a_fin):
    a = np.asarray(a_fin, dtype=float)
    diffs2 = (a.reshape(-1, 1) - a.reshape(1, -1)) ** 2
    return float(np.mean(np.sum(P * diffs2, axis=1)))


def run_episode_fixed(env, gamma_fin=5.0):
    """
    Fixed network: no rewiring, fixed financial policy.
    """
    env.reset()

    rets, risk20 = [], []
    var_b, gini_infl, coord = [], [], []
    mean_abs_a = []

    while True:
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)
        mean_abs_a.append(float(np.mean(np.abs(a_fin))))

        _, _, done, info = env.step(a_fin, None, None)

        rets.append(info["return"])
        risk20.append(info["risk_v"])
        var_b.append(info["belief_var"])

        infl = env.P.sum(axis=0)
        gini_infl.append(gini(infl))
        coord.append(coordination_metric(env.P, a_fin))

        if done:
            break

    return summarize_episode(env, rets, risk20, var_b, gini_infl, coord, mean_abs_a)


def run_episode_similarity_rewire(env, gamma_fin=5.0):
    """
    Belief-similarity rewiring: every net_period, connect to closest beliefs.
    Financial policy stays fixed.
    """
    env.reset()

    rets, risk20 = [], []
    var_b, gini_infl, coord = [], [], []
    mean_abs_a = []

    while True:
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)
        mean_abs_a.append(float(np.mean(np.abs(a_fin))))

        # Build rewiring action if it's a rewiring day
        net_logits = None
        net_w_logits = None

        if (env.t % env.net_period) == 0:
            b = env.b.copy()
            net_logits = np.zeros((env.N, env.N), dtype=np.float32)
            for i in range(env.N):
                dist = np.abs(b[i] - b)
                score = -dist
                score[i] = -np.inf
                net_logits[i] = score
            net_w_logits = np.zeros((env.N, env.K), dtype=np.float32)

        _, _, done, info = env.step(a_fin, net_logits, net_w_logits)

        rets.append(info["return"])
        risk20.append(info["risk_v"])
        var_b.append(info["belief_var"])

        infl = env.P.sum(axis=0)
        gini_infl.append(gini(infl))
        coord.append(coordination_metric(env.P, a_fin))

        if done:
            break

    return summarize_episode(env, rets, risk20, var_b, gini_infl, coord, mean_abs_a)


@torch.no_grad()
def run_episode_learned(env, actor, gamma_fin=5.0, device="cpu"):
    """
    Learned network policy:
    - At rewiring days, actor proposes net action
    - Financial policy remains fixed
    """
    env.reset()

    rets, risk20 = [], []
    var_b, gini_infl, coord = [], [], []
    mean_abs_a = []

    while True:
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)
        mean_abs_a.append(float(np.mean(np.abs(a_fin))))

        net_logits = None
        net_w_logits = None

        if (env.t % env.net_period) == 0:
            s = torch.tensor(build_net_state(env), device=device)
            logits, w_logits = actor(s)
            neighbors, w, _ = sample_network_action(logits, w_logits, env.K)
            net_logits, net_w_logits = build_env_net_inputs(env, neighbors, w)

        _, _, done, info = env.step(a_fin, net_logits, net_w_logits)

        rets.append(info["return"])
        risk20.append(info["risk_v"])
        var_b.append(info["belief_var"])

        infl = env.P.sum(axis=0)
        gini_infl.append(gini(infl))
        coord.append(coordination_metric(env.P, a_fin))

        if done:
            break

    return summarize_episode(env, rets, risk20, var_b, gini_infl, coord, mean_abs_a)


def summarize_episode(env, rets, risk20, var_b, gini_infl, coord, mean_abs_a):
    rets = np.asarray(rets, dtype=float)
    risk20 = np.asarray(risk20, dtype=float)

    vol_proxy = float(np.mean(rets ** 2))
    tail_q05 = float(np.quantile(rets, 0.05))
    cumret_min = float(np.min(np.cumsum(rets)))

    return {
        "vol_proxy_mean_R2": vol_proxy,
        "risk20_mean": float(np.mean(risk20)),
        "risk20_max": float(np.max(risk20)),
        "tail_return_q05": tail_q05,
        "cumret_min_proxy": cumret_min,
        "coordination_loss_mean": float(np.mean(coord)),
        "influence_gini_mean": float(np.mean(gini_infl)),
        "belief_var_mean": float(np.mean(var_b)),
        "mean_abs_action": float(np.mean(mean_abs_a)),
        "final_price": float(env.p),
    }


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def evaluate_30_seeds(make_env, actor, seeds=range(30), gamma_fin=5.0, device="cpu"):
    """
    Compare:
      [1] Fixed Network
      [2] Similarity Rewire
      [3] PPO Learned Network
    """
    metrics = [
        "vol_proxy_mean_R2",
        "risk20_mean",
        "risk20_max",
        "tail_return_q05",
        "cumret_min_proxy",
        "coordination_loss_mean",
        "influence_gini_mean",
        "belief_var_mean",
        "mean_abs_action",
        "final_price",
    ]

    fixed_vals = {m: [] for m in metrics}
    sim_vals = {m: [] for m in metrics}
    ppo_vals = {m: [] for m in metrics}

    for sd in seeds:
        env1 = make_env(seed=int(sd))
        env2 = make_env(seed=int(sd))
        env3 = make_env(seed=int(sd))

        s_fixed = run_episode_fixed(env1, gamma_fin=gamma_fin)
        s_sim = run_episode_similarity_rewire(env2, gamma_fin=gamma_fin)
        s_ppo = run_episode_learned(env3, actor, gamma_fin=gamma_fin, device=device)

        for m in metrics:
            fixed_vals[m].append(s_fixed[m])
            sim_vals[m].append(s_sim[m])
            ppo_vals[m].append(s_ppo[m])

    print("\n" + "=" * 96)
    print(f"EVALUATION (seeds={len(list(seeds))}) | gamma_fin={gamma_fin}")
    print("=" * 96)
    print(f"{'Metric':24s} | {'Fixed (mean±std)':22s} | {'Sim-Rewire (mean±std)':24s} | {'PPO-Learned (mean±std)':24s}")
    print("-" * 110)

    for m in metrics:
        a_mean, a_std = mean_std(fixed_vals[m])
        b_mean, b_std = mean_std(sim_vals[m])
        c_mean, c_std = mean_std(ppo_vals[m])

        print(
            f"{m:24s} | "
            f"{a_mean: .6g} ± {a_std: .3g}".ljust(22) + " | "
            f"{b_mean: .6g} ± {b_std: .3g}".ljust(24) + " | "
            f"{c_mean: .6g} ± {c_std: .3g}".ljust(24)
        )


# ============================================================
# 10) MAIN
# ============================================================

if __name__ == "__main__":
    # Main stable config (your locked setting)
    def make_env(seed=0):
        return InfoNetworkBondEnv(
            seed=int(seed),
            kappa=0.02,
            horizon=1000,
            phi=0.02,
            net_period=5,
            N=50,
            K=5,
            sigma_eps=0.1,
            tau=0.001
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train network-only PPO
    actor, critic = train_network_only_ppo(
        make_env=make_env,
        iters=200,
        macro_steps_per_iter=64,
        gamma_fin=5.0,
        reward_scale=1e6,
        lr=3e-4,
        device=device,
    )

    # Evaluate vs baselines on 30 seeds
    evaluate_30_seeds(
        make_env=make_env,
        actor=actor,
        seeds=range(30),
        gamma_fin=5.0,
        device=device
    )
