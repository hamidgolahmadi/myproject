# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 15:33:29 2025

@author: hg2e25
"""

# -*- coding: utf-8 -*-
"""
PPO (Network-only) v4 for InfoNetworkBondEnv

What changed vs v3:
- PPO previously found a degenerate solution: high influence concentration (high gini),
  frozen beliefs (low belief_var), but WORSE systemic risk.
- v4 fixes the objective to target the true instability channel in your price model:
    p_{t+1} = p_t + kappa * sum(delta_x) + eps
  Therefore systemic risk is tied to correlated flows (net_flow).

Macro reward (per macro-step = net_period micro-steps):
  R = - mean_risk
      - xi_flow * mean(net_flow^2)
      - eta_gini * gini(in_influence)
      - c_net   * ||P_after - P_before||_1
      - zeta_pol * mean(var(b))      (optional, default 0)

Then scaled by reward_scale.

Also includes:
- stronger critic (mean/std/max aggregation)
- separate lrs + grad clipping
- full 30-seed evaluation:
    Fixed, Similarity-Rewire, PPO-Learned
- ablation runner for reward variants:
    A) risk only
    B) risk + gini
    C) risk + flow2
    D) risk + flow2 + gini + deltaP (+ optional polarization)

Requirements:
- numpy
- torch

IMPORTANT:
- Adjust import line for InfoNetworkBondEnv if your module name differs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 0) IMPORT ENV (adjust as needed)
# ============================================================

from temp import InfoNetworkBondEnv  # <-- change if needed


# ============================================================
# 1) FIXED FINANCIAL POLICY (constant)
# ============================================================

def fixed_financial_policy(env, gamma=5.0):
    """Belief-driven trading, fixed across all experiments."""
    neigh_signal = env.P @ env.b
    return np.tanh(gamma * neigh_signal)


# ============================================================
# 2) HELPERS: gini + coordination + summaries
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


def summarize_episode(env, rets, risk20, var_b, gini_infl, coord, mean_abs_a):
    rets = np.asarray(rets, dtype=float)
    risk20 = np.asarray(risk20, dtype=float)

    return {
        "vol_proxy_mean_R2": float(np.mean(rets ** 2)),
        "risk20_mean": float(np.mean(risk20)),
        "risk20_max": float(np.max(risk20)),
        "tail_return_q05": float(np.quantile(rets, 0.05)),
        "cumret_min_proxy": float(np.min(np.cumsum(rets))),
        "coordination_loss_mean": float(np.mean(coord)),
        "influence_gini_mean": float(np.mean(gini_infl)),
        "belief_var_mean": float(np.mean(var_b)),
        "mean_abs_action": float(np.mean(mean_abs_a)),
        "final_price": float(env.p),
    }


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


# ============================================================
# 3) NETWORK STATE (compact, per-agent)
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
# 4) ACTOR + CRITIC (strong critic)
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


def critic_features(x_agents):
    """Concat(mean, std, max) across agents -> (3d,)"""
    mu = x_agents.mean(dim=0)
    sd = x_agents.std(dim=0)
    mx = x_agents.max(dim=0).values
    return torch.cat([mu, sd, mx], dim=0)


class NetCritic(nn.Module):
    """Centralized critic using (mean, std, max) aggregation."""
    def __init__(self, d_in_agg, hidden=128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(d_in_agg, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_agents):
        xg = critic_features(x_agents)
        return self.v(xg).squeeze(-1)


# ============================================================
# 5) ACTION: SAMPLE TOP-K NEIGHBORS + WEIGHTS
# ============================================================

def sample_topk_without_replacement(probs_row, K):
    """
    Sample K indices without replacement from a probability vector.
    Returns:
      idx_list: list[int]
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
    N = logits.shape[0]
    device = logits.device

    probs = torch.softmax(logits, dim=-1)

    neighbors = torch.zeros((N, K), dtype=torch.long, device=device)
    logp_total = torch.zeros((), device=device)

    for i in range(N):
        p = probs[i].clone()
        p[i] = 0.0
        p = p / (p.sum() + 1e-12)

        idx_list, logp_i = sample_topk_without_replacement(p, K)
        neighbors[i] = torch.tensor(idx_list, dtype=torch.long, device=device)
        logp_total = logp_total + logp_i

    # Normalize log-prob by number of discrete selections (stabilizes PPO ratio)
    logp_total = logp_total / (N * K)

    w = torch.softmax(w_logits, dim=-1)

    return neighbors, w, logp_total



def evaluate_logprob_neighbors(logits, neighbors):
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

    # Normalize log-prob by number of selections
    logp_total = logp_total / (N * K)

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
        net_logits[i, i] = -np.inf
        for k in range(K):
            j = int(neighbors[i, k].item())
            net_logits[i, j] = 0.0

    w_np = w.detach().cpu().numpy().astype(np.float32)
    net_w_logits = np.log(w_np + 1e-12)

    return net_logits, net_w_logits


# ============================================================
# 6) REWARD CONFIG (ablation-ready)
# ============================================================

class RewardConfig:
    """
    Macro reward:
      R = - mean_risk
          - xi_flow * mean(net_flow^2)
          - eta_gini * gini(in_influence)
          - c_net   * ||P_after - P_before||_1
          - zeta_pol * mean(var(b))   (optional)
    All terms are computed over the macro window.
    """
    def __init__(
        self,
        xi_flow=0.0,
        eta_gini=0.0,
        c_net=0.01,
        zeta_pol=0.0,
        reward_scale=1e3,
    ):
        self.xi_flow = float(xi_flow)
        self.eta_gini = float(eta_gini)
        self.c_net = float(c_net)
        self.zeta_pol = float(zeta_pol)
        self.reward_scale = float(reward_scale)


# ============================================================
# 7) MACRO-STEP (SMDP) with improved reward
# ============================================================

@torch.no_grad()
def macro_step(env, actor, critic, gamma_fin=5.0, rcfg=None, device="cpu"):
    """
    One macro-step:
    - Choose network rewiring once
    - Run net_period micro-steps with fixed financial policy
    - Compute macro reward using RewardConfig
    """
    if rcfg is None:
        rcfg = RewardConfig()

    # State at decision time
    s_np = build_net_state(env)
    s = torch.tensor(s_np, device=device)

    # Sample network action
    logits, w_logits = actor(s)
    neighbors, w, logp = sample_network_action(logits, w_logits, env.K)

    # Value
    V = critic(s)

    # Prepare env inputs
    net_logits_np, net_w_logits_np = build_env_net_inputs(env, neighbors, w)

    # Network change cost needs previous P
    P_before = env.P.copy()

    # Roll micro-steps and collect stats
    risk_sum = 0.0
    flow2_sum = 0.0
    pol_sum = 0.0
    steps = 0
    done = False

    for _ in range(env.net_period):
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)

        if (env.t % env.net_period) == 0:
            _, _, done, info = env.step(a_fin, net_logits_np, net_w_logits_np)
        else:
            _, _, done, info = env.step(a_fin, None, None)

        risk_sum += float(info["risk_v"])
        flow_norm = float(info["net_flow"]) / (env.N * env.x_max + 1e-12)
        flow2_sum += flow_norm * flow_norm
        pol_sum += float(info["belief_var"])
        steps += 1

        if done:
            break

    # Macro statistics
    mean_risk = risk_sum / max(steps, 1)
    mean_flow2 = flow2_sum / max(steps, 1)
    mean_pol = pol_sum / max(steps, 1)

    P_after = env.P
    deltaP_L1 = float(np.sum(np.abs(P_after - P_before)))

    infl = P_after.sum(axis=0)
    gini_infl = float(gini(infl))

    # Reward (unscaled)
    R_macro = -mean_risk
    R_macro += -rcfg.xi_flow * mean_flow2
    R_macro += -rcfg.eta_gini * gini_infl
    R_macro += -rcfg.c_net * deltaP_L1
    R_macro += -rcfg.zeta_pol * mean_pol

    # Scale
    R_macro = rcfg.reward_scale * R_macro

    return {
        "s": s,
        "neighbors": neighbors.detach(),
        "logp": logp.detach(),
        "R": torch.tensor(R_macro, dtype=torch.float32, device=device),
        "done": torch.tensor(float(done), dtype=torch.float32, device=device),
        "V": V.detach(),

        # Debug stats (optional)
        "dbg_mean_risk": mean_risk,
        "dbg_mean_flow2": mean_flow2,
        "dbg_gini": gini_infl,
        "dbg_deltaP": deltaP_L1,
        "dbg_mean_pol": mean_pol,
    }


# ============================================================
# 8) ROLLOUT + GAE
# ============================================================

class Rollout:
    def __init__(self):
        self.s = []
        self.neighbors = []
        self.logp = []
        self.R = []
        self.done = []
        self.V = []

        # Optional debug
        self.dbg = []

    def add(self, tr):
        self.s.append(tr["s"])
        self.neighbors.append(tr["neighbors"])
        self.logp.append(tr["logp"])
        self.R.append(tr["R"])
        self.done.append(tr["done"])
        self.V.append(tr["V"])
        self.dbg.append({
            "mean_risk": tr.get("dbg_mean_risk", np.nan),
            "mean_flow2": tr.get("dbg_mean_flow2", np.nan),
            "gini": tr.get("dbg_gini", np.nan),
            "deltaP": tr.get("dbg_deltaP", np.nan),
            "mean_pol": tr.get("dbg_mean_pol", np.nan),
        })

    def stack(self, device="cpu"):
        logp = torch.stack(self.logp).to(device)
        R = torch.stack(self.R).to(device)
        done = torch.stack(self.done).to(device)
        V = torch.stack(self.V).to(device)
        return logp, R, done, V


def compute_returns_advantages(R, done, V, gamma=0.99, lam=0.95):
    """
    Proper GAE(λ) using V_{t+1}.
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
# 9) PPO UPDATE
# ============================================================

def ppo_update(actor, critic, optA, optC, rollout, device="cpu",
               clip=0.2, vf_coef=0.5, gamma=0.99, lam=0.95, epochs=4,
               clip_actor=1.0, clip_critic=0.2):
    """
    PPO update on macro-steps.
    Only neighbor-selection logprob is optimized (weights deterministic).
    """
    logp_old, R, done, V_old = rollout.stack(device=device)
    ret, adv = compute_returns_advantages(R, done, V_old, gamma=gamma, lam=lam)

    T = len(rollout.s)

    for _ in range(epochs):
        actor_loss = 0.0
        critic_loss = 0.0

        for t in range(T):
            s = rollout.s[t].to(device)
            neighbors = rollout.neighbors[t].to(device)

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
        nn.utils.clip_grad_norm_(actor.parameters(), clip_actor)
        optA.step()

        # Update critic
        optC.zero_grad()
        (vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), clip_critic)
        optC.step()

    # Debug aggregates
    dbg_mean_risk = float(np.mean([d["mean_risk"] for d in rollout.dbg]))
    dbg_mean_flow2 = float(np.mean([d["mean_flow2"] for d in rollout.dbg]))
    dbg_mean_gini = float(np.mean([d["gini"] for d in rollout.dbg]))
    dbg_mean_deltaP = float(np.mean([d["deltaP"] for d in rollout.dbg]))

    return {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "R_mean": float(R.detach().cpu().mean().item()),
        "R_std": float(R.detach().cpu().std(unbiased=False).item()),
        "adv_std": float(adv.detach().cpu().std(unbiased=False).item()),
        "dbg_mean_risk": dbg_mean_risk,
        "dbg_mean_flow2": dbg_mean_flow2,
        "dbg_mean_gini": dbg_mean_gini,
        "dbg_mean_deltaP": dbg_mean_deltaP,
    }


# ============================================================
# 10) TRAINING LOOP
# ============================================================

def train_network_only_ppo(
    make_env,
    rcfg,
    iters=200,
    macro_steps_per_iter=64,
    gamma_fin=5.0,
    lr_actor=3e-4,
    lr_critic=3e-4,
    device="cpu",
):
    """
    Trains network-only PPO with macro-step rollouts using RewardConfig.
    """
    env0 = make_env(seed=0)
    d_in = build_net_state(env0).shape[1]
    d_agg = 3 * d_in

    actor = NetActor(d_in=d_in, N=env0.N, K=env0.K, hidden=128).to(device)
    critic = NetCritic(d_in_agg=d_agg, hidden=128).to(device)

    optA = optim.Adam(actor.parameters(), lr=lr_actor)
    optC = optim.Adam(critic.parameters(), lr=lr_critic)

    for it in range(iters):
        env = make_env(seed=int(np.random.randint(0, 10_000)))
        rollout = Rollout()

        for _ in range(macro_steps_per_iter):
            if env.t >= env.horizon:
                env.reset()

            tr = macro_step(
                env, actor, critic,
                gamma_fin=gamma_fin,
                rcfg=rcfg,
                device=device
            )
            rollout.add(tr)

        stats = ppo_update(
            actor, critic, optA, optC, rollout,
            device=device,
            clip=0.2, vf_coef=0.5, gamma=0.99, lam=0.95, epochs=4,
            clip_actor=1.0, clip_critic=0.2
        )

        if (it % 10) == 0:
            print(
                f"iter={it:04d} | "
                f"actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.2f} | "
                f"R_mean={stats['R_mean']:.4g} | R_std={stats['R_std']:.4g} | adv_std={stats['adv_std']:.4g} | "
                f"dbg[risk]={stats['dbg_mean_risk']:.3g} | dbg[flow2]={stats['dbg_mean_flow2']:.3g} | "
                f"dbg[gini]={stats['dbg_mean_gini']:.3g} | dbg[deltaP]={stats['dbg_mean_deltaP']:.3g}"
            )

    return actor, critic


# ============================================================
# 11) EVALUATION (Fixed vs Sim-Rewire vs PPO-Learned)
# ============================================================

def run_episode_fixed(env, gamma_fin=5.0):
    env.reset()
    rets, risk20, var_b, gini_infl, coord, mean_abs_a = [], [], [], [], [], []

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
    env.reset()
    rets, risk20, var_b, gini_infl, coord, mean_abs_a = [], [], [], [], [], []

    while True:
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)
        mean_abs_a.append(float(np.mean(np.abs(a_fin))))

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
    env.reset()
    rets, risk20, var_b, gini_infl, coord, mean_abs_a = [], [], [], [], [], []

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


def evaluate_30_seeds(make_env, actor, seeds=range(30), gamma_fin=5.0, device="cpu"):
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

    print("\n" + "=" * 110)
    print(f"EVALUATION (seeds={len(list(seeds))}) | gamma_fin={gamma_fin}")
    print("=" * 110)
    print(f"{'Metric':24s} | {'Fixed (mean±std)':22s} | {'Sim-Rewire (mean±std)':24s} | {'PPO-Learned (mean±std)':24s}")
    print("-" * 124)

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
# 12) ABLATION RUNNER (optional)
# ============================================================

def run_ablation(make_env, gamma_fin, device, iters=200, macro_steps_per_iter=64, seeds_eval=30):
    """
    Runs a small ablation over reward configs.
    WARNING: This trains multiple PPO policies; it will take longer.
    """
    configs = [
        ("A_risk_only", RewardConfig(xi_flow=0.0, eta_gini=0.0, c_net=0.01, zeta_pol=0.0, reward_scale=1e3)),
        ("B_risk_gini", RewardConfig(xi_flow=0.0, eta_gini=0.5, c_net=0.01, zeta_pol=0.0, reward_scale=1e3)),
        ("C_risk_flow2", RewardConfig(xi_flow=1.0, eta_gini=0.0, c_net=0.01, zeta_pol=0.0, reward_scale=1e3)),
        ("D_all", RewardConfig(xi_flow=1.0, eta_gini=0.5, c_net=0.01, zeta_pol=0.0, reward_scale=1e3)),
    ]

    for name, rcfg in configs:
        print("\n" + "#" * 100)
        print(f"TRAINING ABLATION: {name} | xi_flow={rcfg.xi_flow} | eta_gini={rcfg.eta_gini} | c_net={rcfg.c_net} | zeta_pol={rcfg.zeta_pol}")
        print("#" * 100)

        actor, critic = train_network_only_ppo(
            make_env=make_env,
            rcfg=rcfg,
            iters=iters,
            macro_steps_per_iter=macro_steps_per_iter,
            gamma_fin=gamma_fin,
            lr_actor=3e-4,
            lr_critic=3e-4,
            device=device,
        )

        evaluate_30_seeds(
            make_env=make_env,
            actor=actor,
            seeds=range(seeds_eval),
            gamma_fin=gamma_fin,
            device=device,
        )


# ============================================================
# 13) MAIN
# ============================================================

if __name__ == "__main__":
    # Locked stable env config
    def make_env(seed=0):
        return InfoNetworkBondEnv(
            seed=int(seed),
            kappa=0.02,
            horizon=1000,
            phi=0.02,          # keep your stable setting
            net_period=5,
            N=50,
            K=5,
            sigma_eps=0.1,
            tau=0.001
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gamma_fin = 5.0

    # ---------------------------
    # Main v4 config (recommended)
    # ---------------------------
    # Start conservative:
    # - flow penalty targets herding channel in your price impact model
    # - gini penalty prevents hub domination
    # - small deltaP penalty discourages violent rewiring
    rcfg = RewardConfig(
        xi_flow=1.0,      # penalize correlated flows
        eta_gini=0.5,     # penalize influence concentration
        c_net=0.01,       # penalize large network changes (L1)
        zeta_pol=0.0,     # optional: penalize polarization (keep 0 first)
        reward_scale=1e3
    )

    # Train single policy (v4)
    actor, critic = train_network_only_ppo(
        make_env=make_env,
        rcfg=rcfg,
        iters=200,
        macro_steps_per_iter=64,
        gamma_fin=gamma_fin,
        lr_actor=3e-4,
        lr_critic=3e-4,
        device=device,
    )

    # Evaluate vs baselines
    evaluate_30_seeds(
        make_env=make_env,
        actor=actor,
        seeds=range(30),
        gamma_fin=gamma_fin,
        device=device
    )

    # Optional: ablation (commented by default)
    # run_ablation(make_env, gamma_fin, device, iters=200, macro_steps_per_iter=64, seeds_eval=30)
