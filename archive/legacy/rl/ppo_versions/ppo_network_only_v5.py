# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 18:08:07 2025

@author: hg2e25
"""
 
# -*- coding: utf-8 -*-
"""
PPO (Network-only) v5 for InfoNetworkBondEnv
--------------------------------------------

Key fixes vs v4 (based on your evaluation failure):
1) Penalize HERDING directly via mean_action^2 over the macro window
   (this targets the real instability channel in your price impact model).
2) Much stronger anti-hub pressure: eta_gini = 5.0 (default).
3) Strong network-smoothness: c_net = 0.5 (default) + a HARD constraint:
   each agent can change at most M links per rewiring (M default = 1 or 2).
4) Keep flow2 penalty optional/small; main driver is mean_action^2.
5) Reward scaling kept moderate (default reward_scale=1e3).

You can run:
%runfile '.../ppo_network_only_v5.py' --wdir
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------
# Adjust this import if your env lives elsewhere
# ------------------------------------------------------------
from temp import InfoNetworkBondEnv


# ============================================================
# 1) FIXED FINANCIAL POLICY (constant)
# ============================================================

def fixed_financial_policy(env, gamma=5.0):
    """Belief-driven trading. This stays FIXED across baselines + PPO."""
    neigh_signal = env.P @ env.b
    return np.tanh(gamma * neigh_signal)


# ============================================================
# 2) HELPERS
# ============================================================

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


def mean_std(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return float(np.mean(x)), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


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
# 3) NETWORK STATE (compact, per-agent)
# ============================================================

def build_net_state(env):
    """
    Per-agent macro decision state. Shape: (N, 6)

    Features:
    - b_i
    - (P b)_i
    - in_influence_i
    - global risk_v
    - global var(b)
    - last return
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
# 4) ACTOR + CRITIC
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
        self.head_w = nn.Linear(hidden, K)       # weight logits

    def forward(self, x):
        h = self.backbone(x)
        logits = self.head_logits(h)   # (N,N)
        w_logits = self.head_w(h)      # (N,K)
        return logits, w_logits


def critic_features(x_agents):
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
# 5) DISCRETE ACTION: SAMPLE TOP-K WITHOUT REPLACEMENT
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
      logp:      scalar log-prob (normalized by N*K)
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

    # Normalize log-prob to stabilize PPO ratios (important!)
    logp_total = logp_total / (N * K)

    w = torch.softmax(w_logits, dim=-1)  # (N,K)
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
# 6) HARD CONSTRAINT: MAX M LINK CHANGES PER REWIRE
# ============================================================

def current_neighbors_from_P(P, K):
    """Return (N,K) int array of current neighbors (nonzero in each row)."""
    N = P.shape[0]
    neigh = np.zeros((N, K), dtype=int)
    for i in range(N):
        idx = np.where(P[i] > 0)[0]
        if len(idx) == 0:
            # fallback: choose arbitrary (shouldn't happen with your env init)
            idx = np.array([j for j in range(N) if j != i])[:K]
        # If more than K nonzeros (unlikely), keep top-K by weight
        if len(idx) > K:
            idx = idx[np.argsort(P[i, idx])[-K:]]
        # If fewer than K, pad from largest weights / random
        if len(idx) < K:
            candidates = [j for j in range(N) if j != i and j not in set(idx)]
            pad = np.random.choice(candidates, size=K-len(idx), replace=False)
            idx = np.concatenate([idx, pad])
        neigh[i] = idx[:K]
    return neigh


def enforce_max_changes(prev_neighbors, new_neighbors, M, rng=None):
    """
    Enforce at most M changes per row.
    Strategy:
      - Keep K-M links from prev (random subset)
      - Take up to M links from new that are not in prev
      - If not enough new unique, fill from prev
    """
    if rng is None:
        rng = np.random.default_rng(0)

    prev_neighbors = np.asarray(prev_neighbors, dtype=int)
    new_neighbors = np.asarray(new_neighbors, dtype=int)
    N, K = prev_neighbors.shape

    out = np.zeros((N, K), dtype=int)

    for i in range(N):
        prev = prev_neighbors[i].tolist()
        new = new_neighbors[i].tolist()

        prev_set = set(prev)
        # candidates to add (new links not already present)
        add_candidates = [j for j in new if j not in prev_set]

        m_add = min(M, len(add_candidates))
        add = add_candidates[:m_add]

        # keep K - m_add from prev
        keep_count = K - m_add
        if keep_count <= 0:
            keep = []
        else:
            # random keep to avoid systematic bias
            keep = rng.choice(prev, size=keep_count, replace=False).tolist()

        merged = keep + add

        # If still not K, fill from prev (excluding already used)
        if len(merged) < K:
            fill = [j for j in prev if j not in set(merged)]
            merged += fill[: (K - len(merged))]

        out[i] = np.array(merged[:K], dtype=int)

    return out


# ============================================================
# 7) ENV INPUT BUILDERS
# ============================================================

def build_env_net_inputs(env, neighbors_np, w_np):
    """
    Convert (neighbors, weights) into env inputs:
      net_logits: (N,N) where chosen neighbors have 0, others -1e9
      net_w_logits: (N,K) logits so env softmax -> weights
    """
    N = env.N
    K = env.K

    net_logits = -1e9 * np.ones((N, N), dtype=np.float32)
    for i in range(N):
        net_logits[i, i] = -np.inf
        for k in range(K):
            j = int(neighbors_np[i, k])
            net_logits[i, j] = 0.0

    net_w_logits = np.log(w_np.astype(np.float32) + 1e-12)
    return net_logits, net_w_logits


# ============================================================
# 8) REWARD CONFIG (v5)
# ============================================================

class RewardConfig:
    """
    Macro reward over the macro window:

      R = - mean_risk
          - zeta_herd * mean(mean_action^2)
          - xi_flow   * mean(flow_norm^2)              (optional small)
          - eta_gini  * gini(in_influence)
          - c_net     * ||P_after - P_before||_1
          - zeta_pol  * mean(var(b))                   (optional)

    Then scaled by reward_scale.

    Recommended starting point:
      zeta_herd=2.0, eta_gini=5.0, c_net=0.5, xi_flow=0.1, reward_scale=1e3
      M_changes=1 or 2
    """
    def __init__(
        self,
        zeta_herd=2.0,
        xi_flow=0.1,
        eta_gini=5.0,
        c_net=0.5,
        zeta_pol=0.0,
        reward_scale=1e3,
        M_changes=1,
    ):
        self.zeta_herd = float(zeta_herd)
        self.xi_flow = float(xi_flow)
        self.eta_gini = float(eta_gini)
        self.c_net = float(c_net)
        self.zeta_pol = float(zeta_pol)
        self.reward_scale = float(reward_scale)
        self.M_changes = int(M_changes)


# ============================================================
# 9) MACRO STEP (SMDP)
# ============================================================

@torch.no_grad()
def macro_step(env, actor, critic, gamma_fin, rcfg, device="cpu", rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    # Macro decision state
    s_np = build_net_state(env)
    s = torch.tensor(s_np, device=device)

    # Sample network action
    logits, w_logits = actor(s)
    neighbors_t, w_t, logp = sample_network_action(logits, w_logits, env.K)

    # Critic value at decision time
    V = critic(s)

    # HARD constraint on number of link changes
    prev_neighbors = current_neighbors_from_P(env.P, env.K)  # (N,K)
    new_neighbors = neighbors_t.detach().cpu().numpy().astype(int)
    constrained_neighbors = enforce_max_changes(prev_neighbors, new_neighbors, rcfg.M_changes, rng=rng)

    # Keep weights simple and stable: uniform over selected K
    w_uniform = np.ones((env.N, env.K), dtype=np.float32) / float(env.K)

    net_logits_np, net_w_logits_np = build_env_net_inputs(env, constrained_neighbors, w_uniform)

    # Network change cost needs previous P
    P_before = env.P.copy()

    # Roll micro-steps
    risk_sum = 0.0
    herd_sum = 0.0
    flow2_sum = 0.0
    pol_sum = 0.0
    steps = 0
    done = False

    for _ in range(env.net_period):
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)
        mean_a = float(np.mean(a_fin))

        # Apply rewiring ONLY when env applies it
        if (env.t % env.net_period) == 0:
            _, _, done, info = env.step(a_fin, net_logits_np, net_w_logits_np)
        else:
            _, _, done, info = env.step(a_fin, None, None)

        # Risk term
        risk_sum += float(info["risk_v"])

        # Herding term: mean_action^2
        herd_sum += mean_a * mean_a

        # Optional flow term: normalized net_flow^2
        flow_norm = float(info["net_flow"]) / (env.N * env.x_max + 1e-12)
        flow2_sum += flow_norm * flow_norm

        # Polarization (optional)
        pol_sum += float(info["belief_var"])

        steps += 1
        if done:
            break

    # Macro means
    mean_risk = risk_sum / max(steps, 1)
    mean_herd = herd_sum / max(steps, 1)
    mean_flow2 = flow2_sum / max(steps, 1)
    mean_pol = pol_sum / max(steps, 1)

    P_after = env.P
    deltaP_L1 = float(np.sum(np.abs(P_after - P_before)))

    infl = P_after.sum(axis=0)
    gini_infl = float(gini(infl))

    # Unscaled macro reward (NEGATIVE penalties)
    R_macro = -mean_risk
    R_macro += -rcfg.zeta_herd * mean_herd
    R_macro += -rcfg.xi_flow * mean_flow2
    R_macro += -rcfg.eta_gini * gini_infl
    R_macro += -rcfg.c_net * deltaP_L1
    R_macro += -rcfg.zeta_pol * mean_pol

    R_macro = rcfg.reward_scale * R_macro

    return {
        "s": s,
        "neighbors": torch.tensor(constrained_neighbors, dtype=torch.long, device=device),
        "logp": logp.detach(),
        "R": torch.tensor(R_macro, dtype=torch.float32, device=device),
        "done": torch.tensor(float(done), dtype=torch.float32, device=device),
        "V": V.detach(),
        "dbg": {
            "mean_risk": mean_risk,
            "mean_herd": mean_herd,
            "mean_flow2": mean_flow2,
            "gini": gini_infl,
            "deltaP": deltaP_L1,
            "mean_pol": mean_pol,
        }
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

def ppo_update(actor, critic, optA, optC, rollout, device="cpu",
               clip=0.2, vf_coef=0.5, gamma=0.99, lam=0.95, epochs=4,
               clip_actor=1.0, clip_critic=0.2):
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

        # Actor update
        optA.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), clip_actor)
        optA.step()

        # Critic update
        optC.zero_grad()
        (vf_coef * critic_loss).backward()
        nn.utils.clip_grad_norm_(critic.parameters(), clip_critic)
        optC.step()

    dbg_mean = {
        "risk": float(np.mean([d["mean_risk"] for d in rollout.dbg])),
        "herd": float(np.mean([d["mean_herd"] for d in rollout.dbg])),
        "flow2": float(np.mean([d["mean_flow2"] for d in rollout.dbg])),
        "gini": float(np.mean([d["gini"] for d in rollout.dbg])),
        "deltaP": float(np.mean([d["deltaP"] for d in rollout.dbg])),
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
# 12) TRAINING
# ============================================================

def train_network_only_ppo(make_env, rcfg, iters=200, macro_steps_per_iter=64,
                           gamma_fin=5.0, lr_actor=3e-4, lr_critic=3e-4, device="cpu"):
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

        stats = ppo_update(actor, critic, optA, optC, rollout,
                           device=device, clip=0.2, vf_coef=0.5, gamma=0.99, lam=0.95,
                           epochs=4, clip_actor=1.0, clip_critic=0.2)

        if (it % 10) == 0:
            dbg = stats["dbg"]
            print(
                f"iter={it:04d} | "
                f"actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.2f} | "
                f"R_mean={stats['R_mean']:.3g} | R_std={stats['R_std']:.3g} | adv_std={stats['adv_std']:.4f} | "
                f"dbg[risk]={dbg['risk']:.3g} | dbg[herd]={dbg['herd']:.3g} | dbg[flow2]={dbg['flow2']:.3g} | "
                f"dbg[gini]={dbg['gini']:.3g} | dbg[deltaP]={dbg['deltaP']:.3g}"
            )

    return actor, critic


# ============================================================
# 13) EVALUATION (Fixed vs Sim-Rewire vs PPO-Learned)
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
def run_episode_learned(env, actor, gamma_fin=5.0, device="cpu", M_changes=1):
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
            neighbors, w, _ = sample_network_action(logits, w_logits, env.K)

            prev_neighbors = current_neighbors_from_P(env.P, env.K)
            new_neighbors = neighbors.detach().cpu().numpy().astype(int)
            constrained = enforce_max_changes(prev_neighbors, new_neighbors, M_changes, rng=rng)

            w_uniform = np.ones((env.N, env.K), dtype=np.float32) / float(env.K)
            net_logits, net_w_logits = build_env_net_inputs(env, constrained, w_uniform)

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


def evaluate_30_seeds(make_env, actor, seeds=range(30), gamma_fin=5.0, device="cpu", M_changes=1):
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
        s_ppo = run_episode_learned(env3, actor, gamma_fin=gamma_fin, device=device, M_changes=M_changes)

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
    # ---- env factory (keep consistent with your earlier stable config) ----
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
    gamma_fin = 5.0

    # ---- v5 reward config (start here) ----
    rcfg = RewardConfig(
        zeta_herd=2.0,     # MAIN: penalize herding (mean_action^2)
        xi_flow=0.1,       # small additional flow penalty
        eta_gini=5.0,      # strong anti-hub
        c_net=0.5,         # strong smoothness penalty
        zeta_pol=0.0,      # optional
        reward_scale=1e3,
        M_changes=1        # HARD constraint: at most 1 link change per rewiring
    )

    print("=" * 110)
    print(
        f"CONFIG v5: horizon=1000 | gamma_fin={gamma_fin} | "
        f"zeta_herd={rcfg.zeta_herd} | xi_flow={rcfg.xi_flow} | eta_gini={rcfg.eta_gini} | "
        f"c_net={rcfg.c_net} | M_changes={rcfg.M_changes} | reward_scale={rcfg.reward_scale}"
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
        M_changes=rcfg.M_changes
    )
