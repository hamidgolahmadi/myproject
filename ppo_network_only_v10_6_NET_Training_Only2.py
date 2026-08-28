# -*- coding: utf-8 -*-
"""
train_netppo_baselines_v10_9.py

Unified (single-file) runner for Net-PPO training on 3 baseline topologies:
  - random_fixed
  - scale_free
  - small_world (beta via --beta)

Key features:
- argparse CLI: --topology --beta --seed --topology-seed + core hyperparams
- Safe env construction: passes indeg_cap if InfoNetworkBondEnvFinal supports it
- NetActor temperature on logits + entropy bonus in Net-PPO loss
- Gate is fixed by default (platform/network training only), but options exist.

Requires:
- info_network_env_final.py providing InfoNetworkBondEnvFinal with:
    reset() -> (obs_reg_list, obs_net_list)
    step(g_gate, neighbors=None, w=None) -> ((obs_reg_list, obs_net_list), r_reg, done, info)
"""

import argparse
import math
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from info_network_env_final import InfoNetworkBondEnvFinal


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

LOG_2PI = float(math.log(2.0 * math.pi))

def huber_loss(x: torch.Tensor, delta: float):
    absx = torch.abs(x)
    quad = torch.minimum(absx, torch.tensor(delta, device=x.device))
    lin = absx - quad
    return 0.5 * quad * quad + delta * lin

def compute_gae(rew, done, v, v_next, discount=0.99, lam=0.95):
    """
    rew, done, v: shape (T,)
    v_next: scalar
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


# =============================================================================
# Network sampling with indegree cap (sequential masking)
# =============================================================================

@torch.no_grad()
def masked_sequential_sample_neighbors(logits: torch.Tensor, K: int, indeg_cap: int):
    """
    logits: (N, N)
    returns:
      neighbors: (N, K) long
      logp_total: scalar
    """
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
                # fallback: sample uniformly among allowed
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
            j = int(torch.distributions.Categorical(p).sample().item())
            neighbors[i, k] = j
            logp_total = logp_total + torch.log(p[j] + 1e-12)
            indeg[j] += 1
            p[j] = 0.0

    return neighbors, logp_total

def evaluate_logprob_neighbors_masked(logits: torch.Tensor, neighbors: torch.Tensor, K: int, indeg_cap: int):
    """
    Re-compute sequential-masked log-prob of chosen neighbors.
    logits: (N, N)
    neighbors: (N, K) long
    returns scalar logp_total
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
# Squashed Gaussian gate (kept; default is FIX_GATE mode)
# =============================================================================

def gaussian_logprob(x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * logstd)
    return -0.5 * ((x - mu) ** 2 / (var + 1e-8) + 2.0 * logstd + LOG_2PI)

def sample_squashed_gaussian_gate(mu: torch.Tensor, logstd: torch.Tensor):
    std = torch.exp(logstd)
    z = mu + std * torch.randn_like(mu)
    u = torch.tanh(z)
    gate = 0.5 * (u + 1.0)

    logp_z = gaussian_logprob(z, mu, logstd)
    log_det = torch.log(1.0 - u * u + 1e-6)
    logp_u = logp_z - log_det
    logp_g = logp_u + math.log(2.0)
    logp_gate_norm = logp_g.mean()

    ent = 0.5 * (LOG_2PI + 1.0) + logstd
    ent_gate_mean = ent.mean()
    return gate, logp_gate_norm, ent_gate_mean

def logprob_squashed_gaussian_gate(gate: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
    g = torch.clamp(gate, 1e-6, 1.0 - 1e-6)
    u = 2.0 * g - 1.0
    u = torch.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6)

    z = 0.5 * (torch.log1p(u) - torch.log1p(-u))
    logp_z = gaussian_logprob(z, mu, logstd)
    log_det = torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6)
    logp_u = logp_z - log_det
    logp_g = logp_u + math.log(2.0)
    return logp_g.mean()


# =============================================================================
# Models
# =============================================================================

class GateActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.to_mu = nn.Linear(hidden, 1)
        self.to_logstd_raw = nn.Linear(hidden, 1)

    def forward(self, obs_t: torch.Tensor, logstd_min: float, logstd_max: float):
        h = self.mlp(obs_t)
        mu = self.to_mu(h).squeeze(-1)
        raw = self.to_logstd_raw(h).squeeze(-1)
        s = torch.sigmoid(raw)
        logstd = logstd_min + (logstd_max - logstd_min) * s
        return mu, logstd

class NetActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, emb: int = 64, K: int = 5):
        super().__init__()
        self.K = K
        self.emb = emb
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.to_emb = nn.Linear(hidden, emb)
        self.to_wlogits = nn.Linear(hidden, K)

    def forward(self, obs_t: torch.Tensor, temperature: float = 1.0):
        h = self.mlp(obs_t)
        e = self.to_emb(h)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        base_scale = 1.0 / math.sqrt(e.shape[-1])

        temp = float(temperature)
        logits = temp * (e @ e.t()) * base_scale  # (N, N)

        w_logits = self.to_wlogits(h)  # (N, K)
        return logits, w_logits

class StepCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        pooled = obs_t.mean(dim=0, keepdim=True)
        return self.v(pooled).squeeze(-1)

class MacroCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_t: torch.Tensor):
        pooled = obs_t.mean(dim=0, keepdim=True)
        return self.v(pooled).squeeze(-1)


# =============================================================================
# Platform reward components
# =============================================================================

def build_info_rewards(b_seq: np.ndarray, R_seq: np.ndarray, info_horizon: int):
    """
    b_seq: (T, N) beliefs
    R_seq: (T,) returns
    """
    T, _N = b_seq.shape
    H = int(info_horizon)
    r_info = np.zeros((T,), dtype=float)

    if H < 1:
        return r_info

    for t in range(T):
        t1 = min(t + 1, T - 1)
        t2 = min(t + H, T - 1)
        if t2 <= t1:
            Rbar = float(R_seq[t2])
        else:
            Rbar = float(np.mean(R_seq[t1:t2 + 1]))
        err = b_seq[t] - Rbar
        r_info[t] = - float(np.mean(err * err))
    return r_info

def build_platform_rewards(
    b_seq: np.ndarray,
    R_seq: np.ndarray,
    dP_seq: np.ndarray,
    info_horizon: int,
    alpha: float,
    beta: float,
    gamma: float,
    lambda_dP: float,
):
    r_info = build_info_rewards(b_seq=b_seq, R_seq=R_seq, info_horizon=info_horizon)
    r_pol  = - np.var(b_seq, axis=1)
    r_rew  = - float(lambda_dP) * dP_seq
    r_net = float(alpha) * r_info + float(beta) * r_pol + float(gamma) * r_rew
    return r_net, r_info, r_pol, r_rew


# =============================================================================
# Rollout
# =============================================================================

def rollout_one(env: InfoNetworkBondEnvFinal,
                gate_actor: GateActor,
                net_actor: NetActor,
                step_critic: StepCritic,
                cfg):

    device = torch.device(cfg.device)

    obs_reg, obs_net = env.reset()

    obs_reg_list, obs_net_list = [], []
    done_list = []
    r_reg_list = []

    R_list, b_list, dP_list = [], [], []
    v_step_list = []

    gate_list, logp_gate_list, ent_gate_list = [], [], []
    netmask_list, neighbors_list, w_list, logp_net_list = [], [], [], []

    dbg_last = {}
    net_dbg_last = None

    for _ in range(cfg.rollout_len):
        obs_reg_t = torch.tensor(np.asarray(obs_reg, dtype=np.float32), device=device)
        obs_net_t = torch.tensor(np.asarray(obs_net, dtype=np.float32), device=device)

        v_step = step_critic(obs_reg_t)

        # Gate
        if cfg.fix_gate:
            gate = torch.full((env.N,), float(cfg.fixed_gate_value), device=device)
            logp_gate_norm = torch.zeros((), device=device)
            ent_gate_mean  = torch.zeros((), device=device)
        else:
            mu, logstd = gate_actor(obs_reg_t, cfg.gate_logstd_min, cfg.gate_logstd_max)
            gate, logp_gate_norm, ent_gate_mean = sample_squashed_gaussian_gate(mu, logstd)

        # Net
        do_net = (env.t % cfg.net_period) == 0
        if do_net:
            logits, w_logits = net_actor(obs_net_t, temperature=cfg.net_temperature)
            neighbors, logp_net = masked_sequential_sample_neighbors(logits, cfg.K, cfg.indeg_cap)
            w = torch.softmax(w_logits, dim=-1)

            # debug stats (row-wise entropy etc.)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                probs = probs.clone()
                probs.fill_diagonal_(0.0)
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                ent_row = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                top1 = torch.max(probs, dim=-1).values
                net_dbg_last = dict(
                    logits_mean=float(logits.mean().item()),
                    logits_std=float(logits.std(unbiased=False).item()),
                    ent_mean=float(ent_row.mean().item()),
                    ent_std=float(ent_row.std(unbiased=False).item()),
                    top1_mean=float(top1.mean().item()),
                    top1_std=float(top1.std(unbiased=False).item()),
                )
        else:
            neighbors = torch.zeros((env.N, cfg.K), dtype=torch.long, device=device)
            logp_net = torch.zeros((), device=device)
            w = torch.zeros((env.N, cfg.K), dtype=torch.float32, device=device)

        # Step env
        if do_net:
            (obs_reg, obs_net), r_reg, done, info = env.step(
                g_gate=gate.detach().cpu().numpy(),
                neighbors=neighbors.detach().cpu().numpy(),
                w=w.detach().cpu().numpy(),
            )
        else:
            (obs_reg, obs_net), r_reg, done, info = env.step(
                g_gate=gate.detach().cpu().numpy(),
                neighbors=None,
                w=None,
            )

        obs_reg_list.append(obs_reg_t)
        obs_net_list.append(obs_net_t)
        v_step_list.append(v_step.squeeze())

        done_list.append(float(done))
        r_reg_list.append(float(r_reg))

        R_list.append(float(info.get("R", 0.0)))
        b_list.append(np.asarray(info.get("b")).copy())
        dP_list.append(float(info.get("dP", 0.0)))

        gate_list.append(gate)
        logp_gate_list.append(logp_gate_norm)
        ent_gate_list.append(ent_gate_mean)

        netmask_list.append(float(do_net))
        neighbors_list.append(neighbors)
        w_list.append(w)
        logp_net_list.append(logp_net)

        dbg_last = info
        if done:
            break

    obs_reg_stack = torch.stack(obs_reg_list)
    obs_net_stack = torch.stack(obs_net_list)

    b_seq = np.stack(b_list, axis=0)
    R_seq = np.asarray(R_list, dtype=float)
    dP_seq = np.asarray(dP_list, dtype=float)

    r_net, r_info, r_pol, r_rew = build_platform_rewards(
        b_seq=b_seq,
        R_seq=R_seq,
        dP_seq=dP_seq,
        info_horizon=cfg.info_horizon,
        alpha=cfg.alpha_info,
        beta=cfg.beta_pol,
        gamma=cfg.gamma_rew,
        lambda_dP=cfg.lambda_dP,
    )

    batch = {
        "obs_reg": obs_reg_stack,
        "obs_net": obs_net_stack,
        "done": torch.tensor(done_list, dtype=torch.float32, device=torch.device(cfg.device)),

        "r_reg": torch.tensor(r_reg_list, dtype=torch.float32, device=torch.device(cfg.device)),
        "r_net": torch.tensor(r_net, dtype=torch.float32, device=torch.device(cfg.device)),
        "r_info": torch.tensor(r_info, dtype=torch.float32, device=torch.device(cfg.device)),
        "r_pol": torch.tensor(r_pol, dtype=torch.float32, device=torch.device(cfg.device)),
        "r_rew": torch.tensor(r_rew, dtype=torch.float32, device=torch.device(cfg.device)),

        "v_step": torch.stack(v_step_list),

        "gate": torch.stack(gate_list),
        "logp_gate": torch.stack(logp_gate_list),
        "ent_gate": torch.stack(ent_gate_list),

        "netmask": torch.tensor(netmask_list, dtype=torch.float32, device=torch.device(cfg.device)),
        "neighbors": neighbors_list,
        "w": w_list,
        "logp_net": torch.stack(logp_net_list),

        "dbg_last": dbg_last,
        "net_dbg_last": net_dbg_last,
    }
    return batch


# =============================================================================
# Macro sequence builder (Net-PPO) — aggregates r_net on net-steps only
# =============================================================================

def build_macro_sequence_from_step_reward(step_rew: torch.Tensor, done: torch.Tensor, netmask: torch.Tensor,
                                         net_period: int, discount: float):
    T = step_rew.shape[0]
    idx = torch.where(netmask > 0.5)[0]
    if idx.numel() == 0:
        return {"idx_net": idx,
                "r_macro": torch.zeros((0,), device=step_rew.device),
                "done_macro": torch.zeros((0,), device=step_rew.device)}

    r_macro = torch.zeros((idx.numel(),), device=step_rew.device)
    done_macro = torch.zeros((idx.numel(),), device=step_rew.device)

    for m, t in enumerate(idx.tolist()):
        s = 0.0
        last = t
        for k in range(net_period):
            tt = t + k
            if tt >= T:
                break
            s += (discount ** k) * float(step_rew[tt].item())
            last = tt
            if done[tt].item() > 0.5:
                break
        r_macro[m] = float(s)
        done_macro[m] = float(done[last].item())

    return {"idx_net": idx, "r_macro": r_macro, "done_macro": done_macro}


# =============================================================================
# Config
# =============================================================================

@dataclass
class ConfigV10_9:
    # topology
    topology: str = "random_fixed"     # random_fixed | scale_free | small_world
    beta_small_world: float = 0.1
    topology_seed: int = 0
    env_seed: int = 0
    train_seed: int = 0

    # env size
    horizon: int = 1000
    N: int = 50
    K: int = 5
    indeg_cap: int = 8
    net_period: int = 5
    gamma_fin: float = 5.0

    # rollout / training
    rollout_len: int = 200
    n_iters: int = 200

    # Platform objective
    info_horizon: int = 10
    alpha_info: float = 10.0
    beta_pol: float = 1.0
    gamma_rew: float = 0.1
    lambda_dP: float = 0.1

    # NetActor controls
    net_temperature: float = 10.0
    net_ent_coef: float = 0.005

    # Gate PPO (optional)
    lr_gate: float = 5e-5
    gate_clip: float = 0.1
    gate_mini_epochs: int = 2
    gate_minibatch: int = 64
    gate_discount: float = 0.99
    gate_gae_lambda: float = 0.95
    gate_ent_coef: float = 0.005
    gate_logstd_min: float = -2.5
    gate_logstd_max: float = -0.5
    w_gate_mean: float = 0.2
    gate_target: float = 0.4

    # Net PPO (platform)
    lr_net: float = 5e-4
    net_clip: float = 0.2
    net_mini_epochs: int = 5
    net_minibatch: int = 32
    net_discount: float = 0.99
    net_gae_lambda: float = 0.95
    net_logp_scale: float = 1.0

    # Critics
    lr_step_critic: float = 5e-4
    lr_macro_critic: float = 5e-4
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    ret_clip_step: float = 20.0
    ret_clip_macro: float = 50.0
    huber_delta: float = 10.0

    # Modes
    train_gate: bool = False
    train_net: bool = True
    fix_gate: bool = True
    fixed_gate_value: float = 0.4

    device: str = "cpu"


# =============================================================================
# Env factory (safe indeg_cap passing)
# =============================================================================

def make_env(cfg: ConfigV10_9) -> InfoNetworkBondEnvFinal:
    kwargs = dict(
        topology=cfg.topology,
        beta_small_world=cfg.beta_small_world,
        topology_seed=cfg.topology_seed,
        seed=cfg.env_seed,
        N=cfg.N,
        K=cfg.K,
        net_period=cfg.net_period,
        horizon=cfg.horizon,
    )

    # Try to pass indeg_cap only if supported
    try:
        sig = inspect.signature(InfoNetworkBondEnvFinal.__init__)
        if "indeg_cap" in sig.parameters:
            kwargs["indeg_cap"] = cfg.indeg_cap
    except Exception:
        # fallback: best effort
        pass

    env = InfoNetworkBondEnvFinal(**kwargs)
    # finance intensity used by env
    try:
        env.gamma_fin = cfg.gamma_fin
    except Exception:
        pass
    return env


# =============================================================================
# Training
# =============================================================================

def train_v10_9(cfg: ConfigV10_9):
    device = torch.device(cfg.device)

    set_seed(cfg.train_seed)
    env = make_env(cfg)

    obs_reg0, obs_net0 = env.reset()
    obs_dim_reg = len(obs_reg0[0])
    obs_dim_net = len(obs_net0[0])

    gate_actor = GateActor(obs_dim=obs_dim_reg, hidden=128).to(device)
    net_actor  = NetActor(obs_dim=obs_dim_net, hidden=128, emb=64, K=cfg.K).to(device)

    step_critic  = StepCritic(obs_dim=obs_dim_reg, hidden=128).to(device)
    macro_critic = MacroCritic(obs_dim=obs_dim_net, hidden=128).to(device)

    opt_gate = optim.Adam(gate_actor.parameters(), lr=cfg.lr_gate)
    opt_net  = optim.Adam(net_actor.parameters(),  lr=cfg.lr_net)

    opt_v_step  = optim.Adam(step_critic.parameters(),  lr=cfg.lr_step_critic)
    opt_v_macro = optim.Adam(macro_critic.parameters(), lr=cfg.lr_macro_critic)

    print("=" * 118)
    print("TRAIN Net-PPO baselines v10.9 (unified)")
    print(f"TOPOLOGY={cfg.topology} beta={cfg.beta_small_world} | topo_seed={cfg.topology_seed} env_seed={cfg.env_seed} train_seed={cfg.train_seed}")
    print(f"N={cfg.N} K={cfg.K} indeg_cap={cfg.indeg_cap} net_period={cfg.net_period} horizon={cfg.horizon} rollout_len={cfg.rollout_len}")
    print(f"Net: temp={cfg.net_temperature} ent_coef={cfg.net_ent_coef} | lr_net={cfg.lr_net} clip={cfg.net_clip} epochs={cfg.net_mini_epochs} mb={cfg.net_minibatch}")
    print(f"Platform reward: alpha*info + beta*(-Var(b)) + gamma*(-lambda_dP*dP)")
    print(f"alpha={cfg.alpha_info} beta={cfg.beta_pol} gamma={cfg.gamma_rew} lambda_dP={cfg.lambda_dP} H={cfg.info_horizon}")
    print(f"MODE: train_net={cfg.train_net} | fix_gate={cfg.fix_gate}({cfg.fixed_gate_value}) train_gate={cfg.train_gate}")
    print("=" * 118)

    for it in range(cfg.n_iters):
        batch = rollout_one(env, gate_actor, net_actor, step_critic, cfg)

        obs_reg = batch["obs_reg"]
        obs_net = batch["obs_net"]
        done = batch["done"]
        T = obs_reg.shape[0]

        # ----------------------------
        # (A) Gate-PPO (optional)
        # ----------------------------
        if cfg.train_gate and (not cfg.fix_gate):
            r_reg = batch["r_reg"]
            v_step = batch["v_step"].detach()
            with torch.no_grad():
                v_next = step_critic(obs_reg[-1]) * (1.0 - done[-1])

            adv_gate, ret_gate = compute_gae(
                rew=r_reg, done=done, v=v_step, v_next=v_next,
                discount=cfg.gate_discount, lam=cfg.gate_gae_lambda,
            )
            adv_gate = (adv_gate - adv_gate.mean()) / (adv_gate.std(unbiased=False) + 1e-8)
            ret_gate = torch.clamp(ret_gate, -cfg.ret_clip_step, cfg.ret_clip_step)

            logp_gate_old = batch["logp_gate"].detach()
            gate_old = batch["gate"].detach()
            idx = torch.arange(T, device=device)

            for _ in range(cfg.gate_mini_epochs):
                perm = idx[torch.randperm(T)]
                for start in range(0, T, cfg.gate_minibatch):
                    mb = perm[start:start + cfg.gate_minibatch]
                    if mb.numel() == 0:
                        continue

                    logp_new_list = []
                    ent_list = []
                    v_mb = torch.zeros((mb.numel(),), device=device)

                    for j, t in enumerate(mb.tolist()):
                        obs_t = obs_reg[t]
                        v_mb[j] = step_critic(obs_t)

                        mu_t, logstd_t = gate_actor(obs_t, cfg.gate_logstd_min, cfg.gate_logstd_max)
                        lp = logprob_squashed_gaussian_gate(gate_old[t], mu_t, logstd_t)
                        logp_new_list.append(lp)

                        ent = (0.5 * (LOG_2PI + 1.0) + logstd_t).mean()
                        ent_list.append(ent)

                    logp_gate_new = torch.stack(logp_new_list)
                    ent_gate = torch.stack(ent_list)

                    ratio = torch.exp(logp_gate_new - logp_gate_old[mb])
                    surr1 = ratio * adv_gate[mb]
                    surr2 = torch.clamp(ratio, 1.0 - cfg.gate_clip, 1.0 + cfg.gate_clip) * adv_gate[mb]
                    pg_loss = -torch.mean(torch.min(surr1, surr2))

                    g_mean_mb = gate_old[mb].mean()
                    gate_mean_pen = cfg.w_gate_mean * (g_mean_mb - cfg.gate_target) ** 2

                    ent_bonus = torch.mean(ent_gate)
                    gate_loss = pg_loss - cfg.gate_ent_coef * ent_bonus + gate_mean_pen

                    td = v_mb - ret_gate[mb]
                    v_loss = torch.mean(huber_loss(td, cfg.huber_delta))

                    opt_gate.zero_grad(set_to_none=True)
                    gate_loss.backward()
                    nn.utils.clip_grad_norm_(gate_actor.parameters(), cfg.max_grad_norm)
                    opt_gate.step()

                    opt_v_step.zero_grad(set_to_none=True)
                    (cfg.vf_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(step_critic.parameters(), cfg.max_grad_norm)
                    opt_v_step.step()

        # ----------------------------
        # (B) Net-PPO on r_net (macro)
        # ----------------------------
        r_net = batch["r_net"]
        netmask = batch["netmask"]

        macro = build_macro_sequence_from_step_reward(
            step_rew=r_net, done=done, netmask=netmask,
            net_period=cfg.net_period, discount=cfg.net_discount
        )
        idx_net = macro["idx_net"]
        M = idx_net.numel()

        net_ratio_means, net_ratio_stds, net_kls, net_clipfracs = [], [], [], []
        vmacro_loss_list, dlogp_list = [], []

        if cfg.train_net and M > 0:
            obs_net_at = obs_net[idx_net]

            with torch.no_grad():
                v_macro = torch.zeros((M,), device=device)
                for m in range(M):
                    v_macro[m] = macro_critic(obs_net_at[m])
                v_macro_next = v_macro[-1] * (1.0 - macro["done_macro"][-1])

            adv_net, ret_net = compute_gae(
                rew=macro["r_macro"],
                done=macro["done_macro"],
                v=v_macro.detach(),
                v_next=v_macro_next.detach(),
                discount=cfg.net_discount,
                lam=cfg.net_gae_lambda,
            )
            adv_net = (adv_net - adv_net.mean()) / (adv_net.std(unbiased=False) + 1e-8)
            ret_net = torch.clamp(ret_net, -cfg.ret_clip_macro, cfg.ret_clip_macro)

            logp_net_old = batch["logp_net"][idx_net].detach() * cfg.net_logp_scale
            neighbors_old = [batch["neighbors"][t] for t in idx_net.tolist()]

            midx = torch.arange(M, device=device)

            for _ in range(cfg.net_mini_epochs):
                perm = midx[torch.randperm(M)]
                for start in range(0, M, cfg.net_minibatch):
                    mbm = perm[start:start + cfg.net_minibatch]
                    if mbm.numel() == 0:
                        continue

                    logp_new_list = []
                    v_mb = torch.zeros((mbm.numel(),), device=device)
                    ent_mb_list = []

                    for j, mm in enumerate(mbm.tolist()):
                        obs_t = obs_net_at[mm]
                        v_mb[j] = macro_critic(obs_t)

                        logits_t, _ = net_actor(obs_t, temperature=cfg.net_temperature)

                        lp = evaluate_logprob_neighbors_masked(
                            logits=logits_t,
                            neighbors=neighbors_old[mm],
                            K=cfg.K,
                            indeg_cap=cfg.indeg_cap,
                        ) * cfg.net_logp_scale
                        logp_new_list.append(lp)

                        # entropy bonus (row-wise, diag removed, renorm)
                        with torch.no_grad():
                            probs = torch.softmax(logits_t, dim=-1)
                            probs = probs.clone()
                            probs.fill_diagonal_(0.0)
                            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
                            ent_row = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                            ent_mb_list.append(ent_row.mean())

                    logp_net_new = torch.stack(logp_new_list)
                    ent_mb = torch.stack(ent_mb_list).mean() if len(ent_mb_list) > 0 else torch.zeros((), device=device)

                    with torch.no_grad():
                        dlogp = torch.mean(torch.abs(logp_net_new - logp_net_old[mbm]))
                        dlogp_list.append(float(dlogp.item()))

                    ratio = torch.exp(logp_net_new - logp_net_old[mbm])
                    surr1 = ratio * adv_net[mbm]
                    surr2 = torch.clamp(ratio, 1.0 - cfg.net_clip, 1.0 + cfg.net_clip) * adv_net[mbm]
                    ppo_loss = -torch.mean(torch.min(surr1, surr2))

                    net_loss = ppo_loss - float(cfg.net_ent_coef) * ent_mb

                    td = v_mb - ret_net[mbm]
                    v_loss = torch.mean(huber_loss(td, cfg.huber_delta))

                    opt_net.zero_grad(set_to_none=True)
                    net_loss.backward()
                    nn.utils.clip_grad_norm_(net_actor.parameters(), cfg.max_grad_norm)
                    opt_net.step()

                    opt_v_macro.zero_grad(set_to_none=True)
                    (cfg.vf_coef * v_loss).backward()
                    nn.utils.clip_grad_norm_(macro_critic.parameters(), cfg.max_grad_norm)
                    opt_v_macro.step()

                    with torch.no_grad():
                        approx_kl = torch.mean(logp_net_old[mbm] - logp_net_new)
                        clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.net_clip).float())

                    net_ratio_means.append(float(ratio.mean().item()))
                    net_ratio_stds.append(float(ratio.std(unbiased=False).item()))
                    net_kls.append(float(approx_kl.item()))
                    net_clipfracs.append(float(clipfrac.item()))
                    vmacro_loss_list.append(float(v_loss.item()))

        # ----------------------------
        # Prints
        # ----------------------------
        if it % cfg.print_every == 0:
            dbg = batch["dbg_last"]
            nd = batch.get("net_dbg_last", None) or {}

            rnet_mean = float(batch["r_net"].mean().item())
            rinfo_mean = float(batch["r_info"].mean().item())
            rpol_mean = float(batch["r_pol"].mean().item())
            rrew_mean = float(batch["r_rew"].mean().item())

            net_ratio_m = float(np.mean(net_ratio_means)) if len(net_ratio_means) else 1.0
            net_ratio_s = float(np.mean(net_ratio_stds))  if len(net_ratio_stds)  else 0.0
            mean_dlogp = float(np.mean(dlogp_list)) if len(dlogp_list) else 0.0

            print(
                f"iter={it:04d} | "
                f"r_reg={float(batch['r_reg'].mean().item()):+.4f}±{float(batch['r_reg'].std(unbiased=False).item()):.4f} | "
                f"r_net={rnet_mean:+.4f} (info={rinfo_mean:+.4f}, pol={rpol_mean:+.4f}, rew={rrew_mean:+.4f}) | "
                f"NET(M={M}): ratio={net_ratio_m:.3f}±{net_ratio_s:.3f} "
                f"kl={np.mean(net_kls) if len(net_kls) else 0.0:+.4f} clip={np.mean(net_clipfracs) if len(net_clipfracs) else 0.0:.3f} "
                f"dlogp={mean_dlogp:.4f} | "
                f"logits_mu={nd.get('logits_mean',0.0):+.3f} logits_std={nd.get('logits_std',0.0):.3f} "
                f"ent={nd.get('ent_mean',0.0):.3f}±{nd.get('ent_std',0.0):.3f} "
                f"top1={nd.get('top1_mean',0.0):.4f}±{nd.get('top1_std',0.0):.4f} | "
                f"gate={float(batch['gate'].mean().item()):.3f}±{float(batch['gate'].std(unbiased=False).item()):.3f} | "
                f"dbg[riskS]={dbg.get('riskS',0.0):.3g} dbg[flow2]={dbg.get('flow2',0.0):.3g} dbg[gini]={dbg.get('gini',0.0):.3g} "
                f"dbg[dP]={dbg.get('dP',0.0):.3g} dbg[g_mean]={dbg.get('g_mean',0.0):.3g}"
            )


# add print_every without polluting earlier dataclass block
setattr(ConfigV10_9, "print_every", 10)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Net-PPO trainer on baseline topologies (random_fixed, scale_free, small_world)."
    )

    p.add_argument("--topology", type=str, default="random_fixed",
                   choices=["random_fixed", "scale_free", "small_world"],
                   help="Network topology baseline.")
    p.add_argument("--beta", type=float, default=0.1,
                   help="Small-world rewiring beta (only used when topology=small_world).")

    p.add_argument("--seed", type=int, default=0,
                   help="Convenience seed: sets env_seed, topology_seed, train_seed unless you override individually.")
    p.add_argument("--env-seed", type=int, default=None, help="Seed for environment stochastic process.")
    p.add_argument("--topology-seed", type=int, default=None, help="Seed for topology construction.")
    p.add_argument("--train-seed", type=int, default=None, help="Seed for torch/numpy training RNG.")

    # key sizes
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--indeg-cap", type=int, default=8)
    p.add_argument("--net-period", type=int, default=5)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--rollout-len", type=int, default=200)
    p.add_argument("--n-iters", type=int, default=200)

    # net behavior
    p.add_argument("--net-temperature", type=float, default=10.0)
    p.add_argument("--net-ent-coef", type=float, default=0.005)

    # platform reward weights
    p.add_argument("--info-horizon", type=int, default=10)
    p.add_argument("--alpha-info", type=float, default=10.0)
    p.add_argument("--beta-pol", type=float, default=1.0)
    p.add_argument("--gamma-rew", type=float, default=0.1)
    p.add_argument("--lambda-dP", type=float, default=0.1)

    # gate mode
    p.add_argument("--fix-gate", action="store_true", default=True)
    p.add_argument("--fixed-gate", type=float, default=0.4)

    # device + printing
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--print-every", type=int, default=10)

    return p.parse_args()

def main():
    args = parse_args()

    # seed wiring
    base_seed = int(args.seed)
    env_seed = base_seed if args.env_seed is None else int(args.env_seed)
    topo_seed = base_seed if args.topology_seed is None else int(args.topology_seed)
    train_seed = base_seed if args.train_seed is None else int(args.train_seed)

    cfg = ConfigV10_9(
        topology=args.topology,
        beta_small_world=float(args.beta),
        topology_seed=topo_seed,
        env_seed=env_seed,
        train_seed=train_seed,

        N=int(args.N),
        K=int(args.K),
        indeg_cap=int(args.indeg_cap),
        net_period=int(args.net_period),
        horizon=int(args.horizon),
        rollout_len=int(args.rollout_len),
        n_iters=int(args.n_iters),

        net_temperature=float(args.net_temperature),
        net_ent_coef=float(args.net_ent_coef),

        info_horizon=int(args.info_horizon),
        alpha_info=float(args.alpha_info),
        beta_pol=float(args.beta_pol),
        gamma_rew=float(args.gamma_rew),
        lambda_dP=float(args.lambda_dP),

        fix_gate=bool(args.fix_gate),
        fixed_gate_value=float(args.fixed_gate),

        device=str(args.device),
    )
    cfg.print_every = int(args.print_every)

    # sanity
    if cfg.topology != "small_world":
        cfg.beta_small_world = float(args.beta)  # harmless; kept for consistent prints

    train_v10_9(cfg)

if __name__ == "__main__":
    main()
