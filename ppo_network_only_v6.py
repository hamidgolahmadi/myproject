# -*- coding: utf-8 -*-
"""
PPO (Network-only) v6 for InfoNetworkBondEnv
--------------------------------------------

What v6 fixes (hard, structural fixes):
1) NO arbitrary reward_scale. Rewards are made dimensionless by explicit normalization.
2) Hard anti-hub constraint via INDEGREE CAPACITY (each node can be chosen by at most C agents).
3) Penalize FLOW SPIKES (4th moment) rather than only mean(flow)^2 (which was redundant with flow2).
4) Network smoothness penalty uses normalized deltaP: deltaP_L1 / (2N)
5) Log-prob normalization by (N*K) is preserved.
6) Deterministic uniform weights on selected neighbors for stability (you can learn weights later).

Run:
%runfile '.../ppo_network_only_v6.py' --wdir

Notes:
- This script expects your InfoNetworkBondEnv class in temp.py (same as you used).
- If your env signature differs, adjust make_env() at bottom accordingly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------
# Adjust import if needed
# ------------------------------------------------------------
from temp import InfoNetworkBondEnv


# ============================================================
# 1) FIXED FINANCIAL POLICY (constant, identical across baselines)
# ============================================================

def fixed_financial_policy(env, gamma=5.0):
    """Belief-driven trading (fixed rule)."""
    neigh_signal = env.P @ env.b
    return np.tanh(gamma * neigh_signal)


# ============================================================
# 2) METRICS / HELPERS
# ============================================================

def mean_std(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return float(np.mean(x)), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


def gini(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s < eps:
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


# ============================================================
# 3) NETWORK STATE (compact per-agent features)
# ============================================================

def build_net_state(env):
    """
    Per-agent macro decision state. Shape: (N, 6)

    Features:
      0) b_i
      1) (P b)_i
      2) in_influence_i (sum over column i)
      3) global risk_v
      4) global var(b)
      5) last return
    """
    b = env.b
    m = env.P @ env.b
    infl_in = env.P.sum(axis=0)

    risk_v = float(env.risk_v)
    var_b = float(np.var(env.b))
    R_prev = float(env.R_prev)

    N = env.N
    x = np.zeros((N, 6), dtype=np.float32)
    x[:, 0] = b
    x[:, 1] = m
    x[:, 2] = infl_in
    x[:, 3] = risk_v
    x[:, 4] = var_b
    x[:, 5] = R_prev
    return x


# ============================================================
# 4) ACTOR + CRITIC (network-only)
# ============================================================

class NetActor(nn.Module):
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
        self.head_logits = nn.Linear(hidden, N)  # neighbor logits
        self.head_w = nn.Linear(hidden, K)       # weight logits (unused for now)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.head_logits(h)   # (N,N)
        w_logits = self.head_w(h)      # (N,K)
        return logits, w_logits


def critic_features(x_agents):
    """Global aggregation: mean / std / max over agents -> (3*d_in,)"""
    mu = x_agents.mean(dim=0)
    sd = x_agents.std(dim=0)
    mx = x_agents.max(dim=0).values
    return torch.cat([mu, sd, mx], dim=0)


class NetCritic(nn.Module):
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
# 5) TOP-K WITHOUT REPLACEMENT (stochastic)
# ============================================================

def sample_topk_without_replacement(probs_row, K):
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
    Returns:
      neighbors: (N,K) LongTensor
      w:         (N,K) simplex weights (softmax)
      logp:      scalar log-prob normalized by (N*K)
    """
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

    logp_total = logp_total / (N * K)

    # Weight head is kept but we keep deterministic uniform weights for now
    _ = torch.softmax(w_logits, dim=-1)
    w = torch.ones((N, K), device=device) / float(K)

    return neighbors, w, logp_total


def evaluate_logprob_neighbors(logits, neighbors):
    """
    Log-prob of the same chosen neighbors under current logits.
    Same without-replacement approximation as sampling.
    Returns normalized scalar logp.
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

    logp_total = logp_total / (N * K)
    return logp_total


# ============================================================
# 6) HARD CONSTRAINT: INDEGREE CAPACITY (anti-hub)
# ============================================================

def current_neighbors_from_P(P, K):
    N = P.shape[0]
    neigh = np.zeros((N, K), dtype=int)
    for i in range(N):
        idx = np.where(P[i] > 0)[0]
        if len(idx) == 0:
            idx = np.array([j for j in range(N) if j != i])[:K]
        if len(idx) > K:
            idx = idx[np.argsort(P[i, idx])[-K:]]
        if len(idx) < K:
            candidates = [j for j in range(N) if j != i and j not in set(idx)]
            pad = np.random.choice(candidates, size=K - len(idx), replace=False)
            idx = np.concatenate([idx, pad])
        neigh[i] = idx[:K]
    return neigh


def enforce_max_changes(prev_neighbors, proposed_neighbors, M, rng):
    prev_neighbors = np.asarray(prev_neighbors, dtype=int)
    proposed_neighbors = np.asarray(proposed_neighbors, dtype=int)
    N, K = prev_neighbors.shape
    out = np.zeros((N, K), dtype=int)

    for i in range(N):
        prev = prev_neighbors[i].tolist()
        prop = proposed_neighbors[i].tolist()
        prev_set = set(prev)

        add_candidates = [j for j in prop if j not in prev_set]
        m_add = min(M, len(add_candidates))
        add = add_candidates[:m_add]

        keep_count = K - m_add
        keep = rng.choice(prev, size=keep_count, replace=False).tolist() if keep_count > 0 else []

        merged = keep + add
        if len(merged) < K:
            fill = [j for j in prev if j not in set(merged)]
            merged += fill[: (K - len(merged))]

        out[i] = np.array(merged[:K], dtype=int)

    return out


def enforce_indegree_capacity(neighbors, C, rng):
    """
    Enforce that each node j can be selected at most C times across all i.
    Greedy procedure:
      - Flatten proposed edges with scores based on their within-row ranking
      - Keep edges while respecting:
          (a) each row i has exactly K edges
          (b) each column j indegree <= C
    Inputs:
      neighbors: (N,K) proposed integer neighbors
      C: capacity for indegree
    Output:
      neighbors_cap: (N,K) feasible neighbors
    """
    neighbors = np.asarray(neighbors, dtype=int)
    N, K = neighbors.shape

    # Edge candidates with a simple "rank score" (earlier in list = better)
    edges = []
    for i in range(N):
        for r in range(K):
            j = int(neighbors[i, r])
            score = float(K - r)  # higher is better
            edges.append((score, i, j))

    # Sort descending by score
    edges.sort(key=lambda x: x[0], reverse=True)

    indeg = np.zeros(N, dtype=int)
    chosen = [[] for _ in range(N)]

    for _, i, j in edges:
        if len(chosen[i]) >= K:
            continue
        if indeg[j] >= C:
            continue
        if j == i:
            continue
        if j in chosen[i]:
            continue
        chosen[i].append(j)
        indeg[j] += 1

    # Fill any missing slots per row with random feasible choices
    for i in range(N):
        if len(chosen[i]) < K:
            needed = K - len(chosen[i])
            # Feasible candidates: not self, not already chosen, and indeg < C
            candidates = [j for j in range(N)
                          if j != i and (j not in chosen[i]) and (indeg[j] < C)]
            # If capacity is tight, relax by ignoring capacity (last resort)
            if len(candidates) < needed:
                candidates = [j for j in range(N) if j != i and (j not in chosen[i])]
            pick = rng.choice(candidates, size=needed, replace=False).tolist()
            for j in pick:
                if indeg[j] < C:
                    indeg[j] += 1
                chosen[i].append(int(j))

    out = np.zeros((N, K), dtype=int)
    for i in range(N):
        out[i] = np.array(chosen[i][:K], dtype=int)
    return out


# ============================================================
# 7) ENV INPUT BUILDERS
# ============================================================

def build_env_net_inputs(env, neighbors_np):
    """
    Convert chosen neighbors into env inputs:
      net_logits: (N,N) chosen neighbors have 0, others -1e9
      net_w_logits: (N,K) zeros -> uniform weights under env softmax
    """
    N, K = env.N, env.K
    net_logits = -1e9 * np.ones((N, N), dtype=np.float32)
    for i in range(N):
        net_logits[i, i] = -np.inf
        for k in range(K):
            j = int(neighbors_np[i, k])
            net_logits[i, j] = 0.0

    # zeros => softmax => uniform weights
    net_w_logits = np.zeros((N, K), dtype=np.float32)
    return net_logits, net_w_logits


# ============================================================
# 8) REWARD CONFIG (dimensionless)
# ============================================================

class RewardConfig:
    """
    Dimensionless macro reward (no arbitrary reward_scale):

      risk_scaled   = mean_risk / risk_unit
      flow2_mean    = mean( (mean_action)^2 )
      flow4_mean    = mean( (mean_action)^4 )   # spike penalty
      deltaP_norm   = deltaP_L1 / (2N)
      gini_infl     = gini(in_influence)

      R = - w_risk * risk_scaled
          - w_flow2 * flow2_mean
          - w_flow4 * flow4_mean
          - w_gini * gini_infl
          - w_net  * deltaP_norm

    risk_unit should be ~ typical mean_risk magnitude to bring risk_scaled ~ O(1).
    """
    def __init__(
        self,
        risk_unit=1e-6,
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=2.0,
        w_net=1.0,
        M_changes=1,
        indeg_cap=10,
    ):
        self.risk_unit = float(risk_unit)
        self.w_risk = float(w_risk)
        self.w_flow2 = float(w_flow2)
        self.w_flow4 = float(w_flow4)
        self.w_gini = float(w_gini)
        self.w_net = float(w_net)
        self.M_changes = int(M_changes)
        self.indeg_cap = int(indeg_cap)


# ============================================================
# 9) MACRO STEP (SMDP)
# ============================================================

@torch.no_grad()
def macro_step(env, actor, critic, gamma_fin, rcfg, device="cpu", rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    # State
    s_np = build_net_state(env)
    s = torch.tensor(s_np, device=device)

    logits, w_logits = actor(s)
    neighbors_t, _, logp = sample_network_action(logits, w_logits, env.K)
    V = critic(s)

    prev_neighbors = current_neighbors_from_P(env.P, env.K)
    proposed = neighbors_t.detach().cpu().numpy().astype(int)

    # Limit per-agent changes first
    constrained = enforce_max_changes(prev_neighbors, proposed, rcfg.M_changes, rng=rng)

    # Enforce global indegree capacity (anti-hub)
    constrained = enforce_indegree_capacity(constrained, rcfg.indeg_cap, rng=rng)

    net_logits_np, net_w_logits_np = build_env_net_inputs(env, constrained)

    P_before = env.P.copy()

    risk_sum = 0.0
    flow2_sum = 0.0
    flow4_sum = 0.0
    steps = 0
    done = False

    for _ in range(env.net_period):
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)
        mean_a = float(np.mean(a_fin))

        if (env.t % env.net_period) == 0:
            _, _, done, info = env.step(a_fin, net_logits_np, net_w_logits_np)
        else:
            _, _, done, info = env.step(a_fin, None, None)

        risk_sum += float(info["risk_v"])
        flow2_sum += mean_a * mean_a
        flow4_sum += (mean_a * mean_a) ** 2

        steps += 1
        if done:
            break

    mean_risk = risk_sum / max(steps, 1)
    mean_flow2 = flow2_sum / max(steps, 1)
    mean_flow4 = flow4_sum / max(steps, 1)

    P_after = env.P
    deltaP_L1 = float(np.sum(np.abs(P_after - P_before)))
    deltaP_norm = deltaP_L1 / (2.0 * env.N + 1e-12)

    infl = P_after.sum(axis=0)
    gini_infl = float(gini(infl))

    risk_scaled = mean_risk / (rcfg.risk_unit + 1e-12)

    R_macro = 0.0
    R_macro += -rcfg.w_risk * risk_scaled
    R_macro += -rcfg.w_flow2 * mean_flow2
    R_macro += -rcfg.w_flow4 * mean_flow4
    R_macro += -rcfg.w_gini * gini_infl
    R_macro += -rcfg.w_net * deltaP_norm

    return {
        "s": s,
        "neighbors": torch.tensor(constrained, dtype=torch.long, device=device),
        "logp": logp.detach(),
        "R": torch.tensor(R_macro, dtype=torch.float32, device=device),
        "done": torch.tensor(float(done), dtype=torch.float32, device=device),
        "V": V.detach(),
        "dbg": {
            "risk": mean_risk,
            "risk_scaled": risk_scaled,
            "flow2": mean_flow2,
            "flow4": mean_flow4,
            "gini": gini_infl,
            "deltaP": deltaP_L1,
            "deltaP_norm": deltaP_norm,
            "indeg_cap": rcfg.indeg_cap,
        },
    }


# ============================================================
# 10) ROLLOUT + GAE
# ============================================================

class Rollout:
    def __init__(self):
        self.s = []
        self.neighbors = []
        self.logp = []
        self.R = []
        self.done = []
        self.V = []
        self.dbg = []

    def add(self, tr):
        self.s.append(tr["s"])
        self.neighbors.append(tr["neighbors"])
        self.logp.append(tr["logp"])
        self.R.append(tr["R"])
        self.done.append(tr["done"])
        self.V.append(tr["V"])
        self.dbg.append(tr["dbg"])

    def stack(self, device="cpu"):
        logp = torch.stack(self.logp).to(device)
        R = torch.stack(self.R).to(device)
        done = torch.stack(self.done).to(device)
        V = torch.stack(self.V).to(device)
        return logp, R, done, V


def compute_returns_advantages(R, done, V, gamma=0.99, lam=0.95):
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
# 11) PPO UPDATE
# ============================================================

def ppo_update(
    actor,
    critic,
    optA,
    optC,
    rollout,
    device="cpu",
    clip=0.2,
    vf_coef=0.5,
    gamma=0.99,
    lam=0.95,
    epochs=4,
    clip_actor=1.0,
    clip_critic=0.5,
):
    logp_old, R, done, V_old = rollout.stack(device=device)
    ret, adv = compute_returns_advantages(R, done, V_old, gamma=gamma, lam=lam)

    T = len(rollout.s)

    for _ in range(epochs):
        actor_loss = 0.0
        critic_loss = 0.0

        for t in range(T):
            s = rollout.s[t].to(device)
            neighbors = rollout.neighbors[t].to(device)

            logits, _ = actor(s)
            logp_new = evaluate_logprob_neighbors(logits, neighbors)

            ratio = torch.exp(logp_new - logp_old[t])
            surr1 = ratio * adv[t]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[t]
            actor_loss = actor_loss + (-torch.min(surr1, surr2))

            V = critic(s)
            critic_loss = critic_loss + (ret[t] - V).pow(2)

        actor_loss = actor_loss / T
        critic_loss = critic_loss / T

        optA.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), clip_actor)
        optA.step()

        optC.zero_grad()
        (vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), clip_critic)
        optC.step()

    dbg_mean = {
        "risk": float(np.mean([d["risk"] for d in rollout.dbg])),
        "risk_scaled": float(np.mean([d["risk_scaled"] for d in rollout.dbg])),
        "flow2": float(np.mean([d["flow2"] for d in rollout.dbg])),
        "flow4": float(np.mean([d["flow4"] for d in rollout.dbg])),
        "gini": float(np.mean([d["gini"] for d in rollout.dbg])),
        "deltaP_norm": float(np.mean([d["deltaP_norm"] for d in rollout.dbg])),
    }

    return {
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "R_mean": float(R.detach().cpu().mean().item()),
        "R_std": float(R.detach().cpu().std(unbiased=False).item()),
        "adv_std": float(adv.detach().cpu().std(unbiased=False).item()),
        "dbg": dbg_mean,
    }


# ============================================================
# 12) TRAIN
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
    env0 = make_env(seed=0)
    d_in = build_net_state(env0).shape[1]
    d_agg = 3 * d_in

    actor = NetActor(d_in=d_in, N=env0.N, K=env0.K, hidden=128).to(device)
    critic = NetCritic(d_in_agg=d_agg, hidden=128).to(device)

    optA = optim.Adam(actor.parameters(), lr=lr_actor)
    optC = optim.Adam(critic.parameters(), lr=lr_critic)

    rng = np.random.default_rng(123)

    for it in range(iters):
        env = make_env(seed=int(rng.integers(0, 100000)))
        rollout = Rollout()

        for _ in range(macro_steps_per_iter):
            if env.t >= env.horizon:
                env.reset()

            tr = macro_step(env, actor, critic, gamma_fin=gamma_fin, rcfg=rcfg, device=device, rng=rng)
            rollout.add(tr)

        stats = ppo_update(actor, critic, optA, optC, rollout, device=device)

        if (it % 10) == 0:
            d = stats["dbg"]
            print(
                f"iter={it:04d} | "
                f"actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.2f} | "
                f"R_mean={stats['R_mean']:.4g} | R_std={stats['R_std']:.4g} | adv_std={stats['adv_std']:.4f} | "
                f"dbg[risk]={d['risk']:.3g} | dbg[riskS]={d['risk_scaled']:.3g} | "
                f"dbg[flow2]={d['flow2']:.3g} | dbg[flow4]={d['flow4']:.3g} | "
                f"dbg[gini]={d['gini']:.3g} | dbg[dP]={d['deltaP_norm']:.3g}"
            )

    return actor, critic


# ============================================================
# 13) EVALUATION
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
def run_episode_learned(env, actor, gamma_fin=5.0, device="cpu", M_changes=1, indeg_cap=10):
    env.reset()
    rng = np.random.default_rng(999)

    rets, risk20, var_b, gini_infl, coord, mean_abs_a = [], [], [], [], [], []

    while True:
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)
        mean_abs_a.append(float(np.mean(np.abs(a_fin))))

        net_logits = None
        net_w_logits = None

        if (env.t % env.net_period) == 0:
            s = torch.tensor(build_net_state(env), device=device)
            logits, w_logits = actor(s)
            neighbors, _, _ = sample_network_action(logits, w_logits, env.K)

            prev_neighbors = current_neighbors_from_P(env.P, env.K)
            prop = neighbors.detach().cpu().numpy().astype(int)

            constrained = enforce_max_changes(prev_neighbors, prop, M_changes, rng=rng)
            constrained = enforce_indegree_capacity(constrained, indeg_cap, rng=rng)

            net_logits, net_w_logits = build_env_net_inputs(env, constrained)

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


def evaluate_30_seeds(make_env, actor, seeds=range(30), gamma_fin=5.0, device="cpu", M_changes=1, indeg_cap=10):
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
        s_ppo = run_episode_learned(env3, actor, gamma_fin=gamma_fin, device=device,
                                    M_changes=M_changes, indeg_cap=indeg_cap)

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
# 14) MAIN
# ============================================================

if __name__ == "__main__":
    # IMPORTANT:
    # Your env class in temp.py must match these parameters.
    # If your InfoNetworkBondEnv constructor does NOT have phi, remove it here.
    def make_env(seed=0):
        return InfoNetworkBondEnv(
            seed=int(seed),
            kappa=0.02,
            horizon=1000,
            phi=0.02,       # remove if your env doesn't have it
            net_period=5,
            N=50,
            K=5,
            sigma_eps=0.1,
            tau=0.001
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gamma_fin = 5.0

    # v6: capacity should be >= K and significantly < N to block hubs
    # Typical good start: indeg_cap = 2*K (here 10)
    rcfg = RewardConfig(
        risk_unit=1e-6,   # typical magnitude of mean_risk so risk_scaled ~ O(1)
        w_risk=1.0,
        w_flow2=0.25,
        w_flow4=1.0,
        w_gini=2.0,
        w_net=1.0,
        M_changes=1,
        indeg_cap=10
    )

    print("=" * 110)
    print(
        f"CONFIG v6: horizon=1000 | gamma_fin={gamma_fin} | "
        f"risk_unit={rcfg.risk_unit:g} | "
        f"w_risk={rcfg.w_risk} | w_flow2={rcfg.w_flow2} | w_flow4={rcfg.w_flow4} | "
        f"w_gini={rcfg.w_gini} | w_net={rcfg.w_net} | "
        f"M_changes={rcfg.M_changes} | indeg_cap={rcfg.indeg_cap}"
    )
    print("=" * 110)

    actor, critic = train_network_only_ppo(
        make_env=make_env,
        rcfg=rcfg,
        iters=200,
        macro_steps_per_iter=64,
        gamma_fin=gamma_fin,
        lr_actor=3e-4,
        lr_critic=3e-4,
        device=device
    )

    evaluate_30_seeds(
        make_env=make_env,
        actor=actor,
        seeds=range(30),
        gamma_fin=gamma_fin,
        device=device,
        M_changes=rcfg.M_changes,
        indeg_cap=rcfg.indeg_cap
    )
 