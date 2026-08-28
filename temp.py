# -*- coding: utf-8 -*-
"""
Info-network sandbox (baseline-ready, multi-seed)

Single-file module:
1) Environment: InfoNetworkBondEnv
2) Baselines: fixed network vs belief-similarity rewiring
3) Metrics + episode runner
4) Multi-seed + sweep comparison (mean ± std)
"""

import numpy as np


# ============================================================
# 1) ENVIRONMENT
# ============================================================

class InfoNetworkBondEnv:
    """
    Multi-agent environment with:
    - Information (belief) network P (row-stochastic)
    - Endogenous rewiring every net_period days via top-K selection
    - Financial action: position change in [-1, 1] scaled by x_max
    - Price formation with mean-reversion:
        p_{t+1} = p_t + kappa * net_flow + eps - phi*(p_t - p_star)
    - Belief diffusion (DeGroot + noise):
        b_{t+1} = P_t b_t + eta
    - Systemic risk state: EWMA of R^2 with ~20-day half-life
    """

    def __init__(
        self,
        N=50,
        K=5,
        net_period=5,
        x_max=1,
        p0=100.0,
        kappa=0.02,
        sigma_eps=0.1,
        tau=0.001,
        beta_risk=None,        # default: 20-day half-life
        lambda_risk=1.0,
        alpha_soc=0.1,
        beta_mix=0.5,
        sigma_belief=0.05,
        seed=0,
        horizon=1000,
        phi=0.02,
        p_star=None
    ):
        self.N = int(N)
        self.K = int(K)
        self.net_period = int(net_period)

        self.x_max = float(x_max)
        self.p0 = float(p0)
        self.kappa = float(kappa)
        self.sigma_eps = float(sigma_eps)
        self.tau = float(tau)

        self.phi = float(phi)
        self.p_star = float(p_star) if p_star is not None else self.p0

        # Risk memory ~20-day half-life if not provided
        self.beta_risk = float(beta_risk) if beta_risk is not None else float(2 ** (-1 / 20))

        self.lambda_risk = float(lambda_risk)
        self.alpha_soc = float(alpha_soc)
        self.beta_mix = float(beta_mix)
        self.sigma_belief = float(sigma_belief)

        self.horizon = int(horizon)

        self.rng = np.random.default_rng(seed)
        self.reset()

    @staticmethod
    def _softmax(x):
        """Stable softmax for mapping logits -> simplex weights."""
        x = x - np.max(x)
        e = np.exp(x)
        s = e.sum()
        return e / s if s > 0 else np.ones_like(x) / len(x)

    def _init_P_random_topk(self):
        """Random row-stochastic P with exactly K non-self neighbors per row."""
        P = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            candidates = [j for j in range(self.N) if j != i]
            nbrs = self.rng.choice(candidates, size=self.K, replace=False)
            w = self.rng.random(self.K)
            w /= w.sum()
            P[i, nbrs] = w
        return P

    def reset(self):
        """Reset episode state."""
        self.t = 0

        self.p = self.p0
        self.p_prev = self.p0
        self.R_prev = 0.0

        # Positions + cash
        self.x = self.rng.normal(0.0, 0.1, size=self.N)
        self.c = np.zeros(self.N, dtype=float)

        # Beliefs
        self.b = self.rng.normal(0.0, 0.2, size=self.N)

        # Network
        self.P = self._init_P_random_topk()

        # Risk state (EWMA of squared returns)
        self.risk_v = 0.0

        # Previous actions (for observation / churning awareness)
        self.a_fin_prev = np.zeros(self.N, dtype=float)

        # Credibility memory
        self.in_influence_prev = self.P.sum(axis=0).copy()

        return self._build_obs()

    def _build_obs(self):
        """
        Per-agent observation: local + network summaries + global summaries.
        Returns: list[np.array], length N.
        """
        var_b = float(np.var(self.b))
        mean_a_prev = float(np.mean(self.a_fin_prev))
        R_prev = float(self.R_prev)
        vol20 = float(np.sqrt(max(self.risk_v, 0.0)))

        in_influence = self.P.sum(axis=0)

        obs = []
        for i in range(self.N):
            neigh_belief_mean = float(self.P[i] @ self.b)
            o_i = np.array([
                self.p,                 # price
                self.x[i],              # position
                self.c[i],              # cash
                self.b[i],              # own belief
                self.a_fin_prev[i],     # previous action
                in_influence[i],        # credibility proxy
                neigh_belief_mean,      # perceived consensus belief
                mean_a_prev,            # market pressure proxy
                var_b,                  # belief polarization
                R_prev,                 # last return
                vol20,                  # risk state (20d memory)
            ], dtype=float)
            obs.append(o_i)

        return obs

    def step(self, a_fin, net_logits=None, net_w_logits=None):
        """
        Step one day.

        a_fin: (N,) in [-1,1]
        net_logits: (N,N) used only when t % net_period == 0
        net_w_logits: (N,K) used only when t % net_period == 0
        """
        a_fin = np.clip(np.asarray(a_fin, dtype=float), -1.0, 1.0)

        # ---- network rewiring (slow timescale) ----
        if (self.t % self.net_period) == 0 and (net_logits is not None) and (net_w_logits is not None):
            net_logits = np.asarray(net_logits, dtype=float)
            net_w_logits = np.asarray(net_w_logits, dtype=float)
            P_new = np.zeros_like(self.P)

            for i in range(self.N):
                logits_i = net_logits[i].copy()
                logits_i[i] = -np.inf  # forbid self-loop
                nbrs = np.argsort(logits_i)[-self.K:]
                w = self._softmax(net_w_logits[i])
                P_new[i, nbrs] = w

            self.P = P_new

        # ---- belief update (daily) ----
        eta = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = self.P @ self.b + eta

        # ---- trading ----
        delta_x = a_fin * self.x_max
        self.x = self.x + delta_x
        tc = self.tau * np.abs(delta_x)

        # ---- price formation with mean-reversion ----
        net_flow = float(np.sum(delta_x))
        eps = float(self.rng.normal(0.0, self.sigma_eps))

        self.p_prev = self.p
        p_raw = self.p + self.kappa * net_flow + eps
        p_mr = p_raw - self.phi * (self.p - self.p_star)
        self.p = max(p_mr, 1e-6)

        # ---- return ----
        R = (self.p - self.p_prev) / self.p_prev
        self.R_prev = float(R)

        # ---- risk state ----
        self.risk_v = self.beta_risk * self.risk_v + (1.0 - self.beta_risk) * (R * R)

        # ---- cash update (trade at new price, simplified) ----
        self.c = self.c - self.p * delta_x - tc

        # ---- rewards (not used for baselines, but kept for RL later) ----
        wealth_prev = self.c + self.p_prev * (self.x - delta_x)
        wealth_now = self.c + self.p * self.x
        r_fin = wealth_now - wealth_prev

        in_influence = self.P.sum(axis=0)
        r_cred = in_influence - self.in_influence_prev
        self.in_influence_prev = in_influence.copy()

        diffs = (a_fin.reshape(-1, 1) - a_fin.reshape(1, -1)) ** 2
        r_coord = -np.sum(self.P * diffs, axis=1)

        r_soc = self.beta_mix * r_cred + (1.0 - self.beta_mix) * r_coord
        r_risk = -self.lambda_risk * self.risk_v

        r_total = r_fin + self.alpha_soc * r_soc + r_risk

        # ---- bookkeeping ----
        self.a_fin_prev = a_fin.copy()
        self.t += 1
        done = self.t >= self.horizon

        obs = self._build_obs()
        info = {
            "price": self.p,
            "return": self.R_prev,
            "risk_v": self.risk_v,
            "net_flow": net_flow,
            "mean_action": float(np.mean(a_fin)),
            "belief_var": float(np.var(self.b)),
        }
        return obs, r_total, done, info


# ============================================================
# 2) BASELINES (RULE-BASED)
# ============================================================

def policy_fixed_network(env, gamma=10):
    """
    Baseline A:
    - Network: fixed (no rewiring)
    - Finance: belief-driven trading using neighbor-weighted belief signal
    """
    neigh_signal = env.P @ env.b
    a_fin = np.tanh(gamma * neigh_signal)
    return a_fin, None, None


def policy_belief_similarity_rewire(env, gamma=10, K=None):
    """
    Baseline B:
    - Network: rewires every net_period to K closest-belief agents
    - Finance: same belief-driven trading rule (to isolate network effect)
    """
    K = int(K or env.K)

    neigh_signal = env.P @ env.b
    a_fin = np.tanh(gamma * neigh_signal)

    if (env.t % env.net_period) != 0:
        return a_fin, None, None

    b = env.b.copy()
    net_logits = np.zeros((env.N, env.N), dtype=float)

    for i in range(env.N):
        dist = np.abs(b[i] - b)
        score = -dist
        score[i] = -np.inf
        net_logits[i] = score

    net_w_logits = np.zeros((env.N, K), dtype=float)  # uniform after softmax
    return a_fin, net_logits, net_w_logits


# ============================================================
# 3) METRICS + RUNNER
# ============================================================

def gini(x, eps=1e-12):
    """Gini coefficient for nonnegative vector."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    if x.sum() < eps:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return 1.0 + 1.0 / n - 2.0 * np.sum(cum) / (n * cum[-1] + eps)


def coordination_metric(env, a_fin):
    """Mean_i sum_j P[i,j] (a_i - a_j)^2."""
    a = np.asarray(a_fin, dtype=float)
    diffs2 = (a.reshape(-1, 1) - a.reshape(1, -1)) ** 2
    return float(np.mean(np.sum(env.P * diffs2, axis=1)))


def _mean_std(x):
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x)) if len(x) else 0.0
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return mean, std


def run_episode(env, policy_fn, T=None):
    """Run one episode and return (summary, series)."""
    T = int(T or env.horizon)
    env.reset()

    prices, rets, risk20 = [], [], []
    var_b, gini_infl, coord = [], [], []
    mean_abs_a_list = []       # mean(|a_t|) across agents each step
    mean_abs_dx_list = []      # mean(|delta_x_t|) across agents each step

    for _ in range(T):
        a_fin, net_logits, net_w_logits = policy_fn(env)

        # Record turnover proxies BEFORE env.step mutates anything
        a_fin_arr = np.asarray(a_fin, dtype=float)
        a_fin_arr = np.clip(a_fin_arr, -1.0, 1.0)

        mean_abs_a_list.append(float(np.mean(np.abs(a_fin_arr))))
        mean_abs_dx_list.append(float(np.mean(np.abs(a_fin_arr * env.x_max))))

        _, _, done, info = env.step(a_fin, net_logits, net_w_logits)

        prices.append(info["price"])
        rets.append(info["return"])
        risk20.append(info["risk_v"])
        var_b.append(info["belief_var"])

        in_influence = env.P.sum(axis=0)
        gini_infl.append(gini(in_influence))

        coord.append(coordination_metric(env, a_fin_arr))

        if done:
            break

    rets_arr = np.asarray(rets, dtype=float)
    vol_proxy = float(np.mean(rets_arr ** 2))
    tail_q05 = float(np.quantile(rets_arr, 0.05))
    cumret_min = float(np.min(np.cumsum(rets_arr)))

    risk20_arr = np.asarray(risk20, dtype=float)
    risk20_end = float(risk20_arr[-1]) if len(risk20_arr) else 0.0
    risk20_mean = float(np.mean(risk20_arr)) if len(risk20_arr) else 0.0
    risk20_max = float(np.max(risk20_arr)) if len(risk20_arr) else 0.0

    mean_abs_a = float(np.mean(mean_abs_a_list)) if len(mean_abs_a_list) else 0.0
    mean_abs_dx = float(np.mean(mean_abs_dx_list)) if len(mean_abs_dx_list) else 0.0

    summary = {
        "vol_proxy_mean_R2": vol_proxy,
        "risk20_end": risk20_end,
        "risk20_mean": risk20_mean,
        "risk20_max": risk20_max,
        "tail_return_q05": tail_q05,
        "cumret_min_proxy": cumret_min,

        "coordination_loss_mean": float(np.mean(coord)) if coord else 0.0,
        "influence_gini_mean": float(np.mean(gini_infl)) if gini_infl else 0.0,
        "belief_var_mean": float(np.mean(var_b)) if var_b else 0.0,

        "mean_abs_action": mean_abs_a,     # turnover proxy in action space
        "mean_abs_deltax": mean_abs_dx,    # turnover proxy in position-change space

        "final_price": float(prices[-1]) if prices else env.p0,
    }

    series = {
        "price": prices,
        "return": rets,
        "risk20": risk20,
        "belief_var": var_b,
        "influence_gini": gini_infl,
        "coordination_loss": coord,
        "mean_abs_action": mean_abs_a_list,
        "mean_abs_deltax": mean_abs_dx_list,
    }

    return summary, series


def compare_selected_configs_30seeds(
    seeds=range(30),
    kappa=0.02,
    horizon=1000,
    configs=((5.0, 0.02), (10.0, 0.02)),  # (gamma, phi)
):
    """
    Run only selected stable configs with many seeds.
    Prints mean ± std for each metric for Fixed vs Rewire.
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
        "mean_abs_deltax",
        "final_price",
    ]

    seeds = list(seeds)

    for (gamma, phi) in configs:
        A_vals = {m: [] for m in metrics}  # Fixed
        B_vals = {m: [] for m in metrics}  # Rewire

        for sd in seeds:
            envA = InfoNetworkBondEnv(seed=int(sd), kappa=kappa, horizon=horizon, phi=phi)
            envB = InfoNetworkBondEnv(seed=int(sd), kappa=kappa, horizon=horizon, phi=phi)

            sA, _ = run_episode(envA, lambda e: policy_fixed_network(e, gamma=gamma), T=horizon)
            sB, _ = run_episode(envB, lambda e: policy_belief_similarity_rewire(e, gamma=gamma, K=e.K), T=horizon)

            for m in metrics:
                A_vals[m].append(sA[m])
                B_vals[m].append(sB[m])

        print("\n" + "=" * 84)
        print(f"SELECTED CONFIG: kappa={kappa} | horizon={horizon} | gamma={gamma} | phi={phi} | seeds={len(seeds)}")
        print("=" * 84)
        print(f"{'Metric':24s} | {'Fixed (mean±std)':26s} | {'Rewire (mean±std)':26s}")
        print("-" * 90)

        for m in metrics:
            a_mean, a_std = _mean_std(A_vals[m])
            b_mean, b_std = _mean_std(B_vals[m])
            print(
                f"{m:24s} | "
                f"{a_mean: .6g} ± {a_std: .3g}".ljust(26) + " | "
                f"{b_mean: .6g} ± {b_std: .3g}".ljust(26)
            )


# ============================================================
# 4) MULTI-SEED + SWEEP
# ============================================================

def compare_baselines_multiseed(
    seeds=range(10),
    kappa=0.02,
    horizon=1000,
    gamma_list=(5.0, 10.0),
    phi_list=(0.01, 0.02),
):
    """
    Multi-seed comparison with parameter sweep.
    Prints mean ± std for each metric under each configuration.
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
        "final_price",
    ]

    seeds = list(seeds)

    for gamma in gamma_list:
        for phi in phi_list:
            A_vals = {m: [] for m in metrics}  # Fixed network
            B_vals = {m: [] for m in metrics}  # Rewire

            for sd in seeds:
                envA = InfoNetworkBondEnv(seed=int(sd), kappa=kappa, horizon=horizon, phi=phi)
                envB = InfoNetworkBondEnv(seed=int(sd), kappa=kappa, horizon=horizon, phi=phi)

                sA, _ = run_episode(envA, lambda e: policy_fixed_network(e, gamma=gamma), T=horizon)
                sB, _ = run_episode(envB, lambda e: policy_belief_similarity_rewire(e, gamma=gamma, K=e.K), T=horizon)

                for m in metrics:
                    A_vals[m].append(sA[m])
                    B_vals[m].append(sB[m])

            print("\n" + "=" * 78)
            print(f"CONFIG: kappa={kappa} | horizon={horizon} | gamma={gamma} | phi={phi} | seeds={len(seeds)}")
            print("=" * 78)
            print(f"{'Metric':24s} | {'Fixed (mean±std)':26s} | {'Rewire (mean±std)':26s}")
            print("-" * 84)

            for m in metrics:
                a_mean, a_std = _mean_std(A_vals[m])
                b_mean, b_std = _mean_std(B_vals[m])
                print(
                    f"{m:24s} | "
                    f"{a_mean: .6g} ± {a_std: .3g}".ljust(26) + " | "
                    f"{b_mean: .6g} ± {b_std: .3g}".ljust(26)
                )


# ============================================================
# 5) ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    compare_selected_configs_30seeds(
        seeds=range(30),
        kappa=0.02,
        horizon=1000,
        configs=((5.0, 0.02), (10.0, 0.02)),
    )
