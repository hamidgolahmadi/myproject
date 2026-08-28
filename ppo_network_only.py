# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 18:24:10 2025

@author: hg2e25
"""

# -*- coding: utf-8 -*-
"""
PPO (Network-only) for InfoNetworkBondEnv

Path A:
- Financial policy is fixed (rule-based): a_fin = tanh(gamma * (P @ b))
- PPO learns ONLY network action (rewiring every net_period days)
- Macro-step (SMDP): one PPO step corresponds to net_period environment steps
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from temp import InfoNetworkBondEnv

# ============================================================
# 0) FIXED FINANCIAL POLICY (kept constant)
# ============================================================

def fixed_financial_policy(env, gamma=5.0):
    """Return a_fin in [-1,1], belief-driven."""
    neigh_signal = env.P @ env.b
    a_fin = np.tanh(gamma * neigh_signal)
    return a_fin


# ============================================================
# 1) STATE FOR NETWORK POLICY (compact, per-agent)
# ============================================================

def build_net_state(env):
    """
    Build per-agent state for network decision at macro-steps.
    Shape: (N, d)

    Features (suggested minimal set):
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
# 2) NETWORK ACTOR + CRITIC (parameter sharing)
# ============================================================

class NetActor(nn.Module):
    """
    Shared actor:
    input: (N, d)
    outputs:
      - logits over neighbors: (N, N)  (for top-K selection)
      - weight logits:         (N, K)  (for simplex weights after env softmax)
    Note: This is O(N^2) output, fine for N~50.
    """

    def __init__(self, d_in, N, K, hidden=128):
        super().__init__()
        self.N = N
        self.K = K

        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Neighbor selection logits per agent
        self.head_logits = nn.Linear(hidden, N)

        # Weight logits for selected K neighbors (env softmax will map to simplex)
        self.head_w = nn.Linear(hidden, K)

    def forward(self, x):
        """
        x: (N, d)
        returns:
          logits:   (N, N)
          w_logits: (N, K)
        """
        h = self.mlp(x)
        logits = self.head_logits(h)
        w_logits = self.head_w(h)
        return logits, w_logits


class NetCritic(nn.Module):
    """
    Centralized critic:
    input: aggregated global embedding (mean over agents) -> scalar V(s)
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
        """
        x_agents: (N, d)
        Aggregate by mean -> (d,)
        """
        xg = x_agents.mean(dim=0)
        return self.v(xg).squeeze(-1)  # scalar


# ============================================================
# 3) ACTION SAMPLING + LOGPROB (hybrid: top-K + weights)
# ============================================================

def sample_network_action(logits, w_logits, K):
    """
    Sample a network action for all agents.
    logits:   (N, N)
    w_logits: (N, K)

    We do:
    - For each agent i: sample K neighbors WITHOUT replacement from categorical probs.
      (Implementation: repeated sampling with masking, simple but OK for small N)
    - For weights: sample from categorical over K slots? (Not needed)
      We treat weights as deterministic softmax(w_logits). For PPO we need logprob.
      Simplest: treat weights as deterministic (no logprob term) in v1.

    Returns:
      neighbors: LongTensor (N, K) indices
      w:         Tensor     (N, K) weights simplex
      logp:      Tensor     scalar total log-prob for neighbor selections
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

        chosen = []
        logp_i = 0.0

        # Sample without replacement (simple loop)
        for k in range(K):
            idx = torch.multinomial(p, num_samples=1).item()
            chosen.append(idx)
            logp_i = logp_i + torch.log(p[idx] + 1e-12)

            # Mask out chosen
            p[idx] = 0.0
            p = p / (p.sum() + 1e-12)

        neighbors[i] = torch.tensor(chosen, device=device)
        logp_total = logp_total + logp_i

    # Deterministic weights (v1): env will softmax them anyway
    w = torch.softmax(w_logits, dim=-1)  # (N,K)

    return neighbors, w, logp_total


def build_env_net_inputs_from_action(env, neighbors, w):
    """
    Convert sampled (neighbors, w) into (net_logits, net_w_logits) for env.step().
    env expects:
      net_logits:   (N, N) where top-K are selected by argsort
      net_w_logits: (N, K) where softmax gives weights

    Trick:
    - We can set net_logits[i,j] = large positive for chosen neighbors, negative otherwise
      so env argsort picks them deterministically.
    - For weights: provide logits as log(w) (up to constant).
    """
    N = env.N
    K = env.K

    net_logits = -1e9 * np.ones((N, N), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            j = int(neighbors[i, k].item())
            net_logits[i, j] = 0.0
        net_logits[i, i] = -np.inf  # forbid self

    w_np = w.detach().cpu().numpy().astype(np.float32)
    net_w_logits = np.log(w_np + 1e-12)  # logits consistent with softmax

    return net_logits, net_w_logits


# ============================================================
# 4) MACRO-STEP ROLLOUT (SMDP)
# ============================================================

@torch.no_grad()
def macro_step(env, actor, critic, gamma_fin=5.0, c_net=0.0, device="cpu"):
    """
    One macro-step = choose network action once, then run env for net_period days.

    Reward design (network-centric):
    - primary: penalize systemic risk (sum over micro-steps)
    - cost: network change penalty (optional, placeholder)
    - cost: turnover penalty (optional, placeholder)

    Returns a transition:
      s, a, logp, R, s_next, done, V(s)
    """
    # Build state at decision time
    s_np = build_net_state(env)               # (N,d)
    s = torch.tensor(s_np, device=device)     # (N,d)

    # Actor outputs
    logits, w_logits = actor(s)
    neighbors, w, logp = sample_network_action(logits, w_logits, env.K)

    # Value estimate
    V = critic(s)

    # Apply macro action (rewire at this step)
    net_logits_np, net_w_logits_np = build_env_net_inputs_from_action(env, neighbors, w)

    # Optional: network change cost (needs previous P snapshot)
    P_before = env.P.copy()

    # Roll micro-steps
    R_macro = 0.0
    done = False

    for _ in range(env.net_period):
        a_fin = fixed_financial_policy(env, gamma=gamma_fin)

        # Only pass network action at the first micro-step (rewire moment)
        if (env.t % env.net_period) == 0:
            _, _, done, info = env.step(a_fin, net_logits_np, net_w_logits_np)
        else:
            _, _, done, info = env.step(a_fin, None, None)

        # Network reward: penalize systemic risk
        R_macro += -float(info["risk_v"])

        if done:
            break

    # Optional: add explicit network-change penalty (L1 distance on P)
    if c_net > 0.0:
        P_after = env.P
        deltaP = np.sum(np.abs(P_after - P_before))
        R_macro += -c_net * float(deltaP)

    # Next state
    s2_np = build_net_state(env)
    s2 = torch.tensor(s2_np, device=device)

    return {
        "s": s,
        "neighbors": neighbors,
        "w": w,
        "logp": logp,
        "R": torch.tensor(R_macro, dtype=torch.float32, device=device),
        "s2": s2,
        "done": torch.tensor(float(done), dtype=torch.float32, device=device),
        "V": V.detach(),
    }


# ============================================================
# 5) PPO BUFFER + UPDATE
# ============================================================

class PPORollout:
    """Minimal rollout storage for macro-steps."""

    def __init__(self):
        self.s = []
        self.logp = []
        self.R = []
        self.done = []
        self.V = []

    def add(self, tr):
        self.s.append(tr["s"])
        self.logp.append(tr["logp"])
        self.R.append(tr["R"])
        self.done.append(tr["done"])
        self.V.append(tr["V"])

    def stack(self, device="cpu"):
        s = self.s
        logp = torch.stack(self.logp).to(device)
        R = torch.stack(self.R).to(device)
        done = torch.stack(self.done).to(device)
        V = torch.stack(self.V).to(device)
        return s, logp, R, done, V


def compute_returns_advantages(R, done, V, gamma=0.99, lam=0.95):
    """
    GAE(λ) on macro-steps.
    R: (T,)
    done: (T,)
    V: (T,)
    returns: (T,), adv: (T,)
    """
    T = R.shape[0]
    adv = torch.zeros_like(R)
    ret = torch.zeros_like(R)

    gae = 0.0
    nextV = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = R[t] + gamma * nextV * mask - V[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        nextV = V[t]
        ret[t] = adv[t] + V[t]

    # Normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return ret, adv


def ppo_update(actor, critic, optA, optC, rollout, clip=0.2, vf_coef=0.5, ent_coef=0.0,
               gamma=0.99, lam=0.95, epochs=4, device="cpu"):
    """
    PPO update (macro-step level).
    NOTE: v1 optimizes only neighbor-selection logprob; weights are deterministic here.
    """
    s_list, logp_old, R, done, V = rollout.stack(device=device)
    ret, adv = compute_returns_advantages(R, done, V, gamma=gamma, lam=lam)

    # We do full-batch epochs (fine for small rollout size)
    for _ in range(epochs):
        # Recompute logp for stored states by re-sampling? (Wrong)
        # Proper PPO needs logp(a|s) of the SAME action.
        # For v1 we store only logp_total from sampling; to recompute we would need
        # to store chosen neighbors and evaluate their logprob under current policy.
        #
        # So for correctness, we implement evaluate_logprob on stored neighbors.
        pass  # replaced below


def evaluate_logprob_neighbors(logits, neighbors):
    """
    Compute log-prob of chosen neighbors under current logits for each agent.
    neighbors: (N,K)
    Returns scalar total log-prob (sum over i,k) with without-replacement approximation.
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
            # Mask chosen for without-replacement approximation
            p[j] = 0.0
            p = p / (p.sum() + 1e-12)

    return logp_total


def ppo_update_correct(actor, critic, optA, optC, rollout,
                       clip=0.2, vf_coef=0.5, ent_coef=0.0,
                       gamma=0.99, lam=0.95, epochs=4, device="cpu"):
    """
    Correct PPO update:
    - Stores s, neighbors, logp_old, R, done, V
    - Recomputes logp_new for the SAME chosen neighbors
    - Uses clipped objective
    """
    # Unpack
    s_list = rollout.s
    neighbors_list = rollout.neighbors  # set below when collecting
    logp_old = torch.stack(rollout.logp).to(device)
    R = torch.stack(rollout.R).to(device)
    done = torch.stack(rollout.done).to(device)
    V_old = torch.stack(rollout.V).to(device)

    ret, adv = compute_returns_advantages(R, done, V_old, gamma=gamma, lam=lam)

    for _ in range(epochs):
        # Actor + Critic losses over all macro-steps (full batch)
        actor_loss = 0.0
        critic_loss = 0.0
        ent_loss = 0.0

        for t, s in enumerate(s_list):
            s = s.to(device)
            neighbors = neighbors_list[t].to(device)

            logits, w_logits = actor(s)
            logp_new = evaluate_logprob_neighbors(logits, neighbors)

            ratio = torch.exp(logp_new - logp_old[t])
            surr1 = ratio * adv[t]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[t]
            actor_loss = actor_loss + (-torch.min(surr1, surr2))

            V = critic(s)
            critic_loss = critic_loss + (ret[t] - V).pow(2)

            # Optional entropy term (approx via categorical entropy over neighbors)
            # We skip proper entropy for without-replacement; keep ent_coef=0 in v1.
            ent_loss = ent_loss + 0.0

        actor_loss = actor_loss / len(s_list)
        critic_loss = critic_loss / len(s_list)

        # Update actor
        optA.zero_grad()
        (actor_loss + ent_coef * ent_loss).backward()
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
    }


# ============================================================
# 6) TRAINING LOOP (macro-step PPO)
# ============================================================

def train_network_only_ppo(
    env_ctor,                 # function returning a fresh env instance
    iters=200,
    macro_steps_per_iter=64,
    gamma_fin=5.0,
    lr=3e-4,
    device="cpu",
):
    """
    Train network policy with PPO using macro-steps.
    env_ctor should return InfoNetworkBondEnv with fixed config:
      gamma_fin=5, phi=0.02 (your main setting), net_period=5, etc.
    """
    # Create a temp env to read sizes
    env = env_ctor()
    d_in = build_net_state(env).shape[1]

    actor = NetActor(d_in=d_in, N=env.N, K=env.K, hidden=128).to(device)
    critic = NetCritic(d_in=d_in, hidden=128).to(device)

    optA = optim.Adam(actor.parameters(), lr=lr)
    optC = optim.Adam(critic.parameters(), lr=lr)

    for it in range(iters):
        env = env_ctor()  # fresh episode env per iteration (simple)
        rollout = PPORollout()
        rollout.neighbors = []  # attach list to store chosen neighbors for PPO eval

        # Collect macro-steps
        for _ in range(macro_steps_per_iter):
            if env.t >= env.horizon:
                env.reset()

            tr = macro_step(env, actor, critic, gamma_fin=gamma_fin, c_net=0.0, device=device)

            # Store transition
            rollout.add(tr)
            rollout.neighbors.append(tr["neighbors"].detach())

        # PPO update
        stats = ppo_update_correct(
            actor, critic, optA, optC, rollout,
            clip=0.2, vf_coef=0.5, ent_coef=0.0,
            gamma=0.99, lam=0.95, epochs=4, device=device
        )

        if (it % 10) == 0:
            print(f"iter={it:04d} | actor_loss={stats['actor_loss']:.4f} | critic_loss={stats['critic_loss']:.4f}")

    return actor, critic


# ============================================================
# 7) EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Import your InfoNetworkBondEnv from your main file if needed.
    # from temp import InfoNetworkBondEnv

    def make_env():
        # Use your chosen stable config
        # NOTE: Adjust constructor params to match your actual env signature.
        return InfoNetworkBondEnv(
            seed=int(np.random.randint(0, 10_000)),
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
    actor, critic = train_network_only_ppo(
        env_ctor=make_env,
        iters=200,
        macro_steps_per_iter=64,
        gamma_fin=5.0,
        lr=3e-4,
        device=device,
    )
