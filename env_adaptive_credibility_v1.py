# env_adaptive_credibility_v1.py
# -------------------------------------------------------------
# Adaptive Credibility Environment (Paper 1)
#
# Key features:
# - Fixed observation network structure P_init
# - Endogenous dynamic influence weights from reputation
# - Reputation updated by EWMA of trading performance
# - Local standardization before softmax
# - Network enters twice by design:
#     (1) belief formation
#     (2) trading signal amplification
# -------------------------------------------------------------

import numpy as np


class InfoNetworkAdaptiveEnv:
    def __init__(
        self,
        P_init,
        seed=0,
        horizon=3000,
        beta=1.0,                 # softmax sensitivity to relative reputation
        gamma=0.9,                # EWMA memory in reputation update
        eps_softmax=1e-8,

        # latent state process
        rho_y=0.985,              # persistence of latent market state
        sigma_y=0.025,            # innovation std of latent state

        # private information
        sigma_signal=0.06,        # std of private signal noise

        # belief formation
        alpha_social=0.75,        # weight on social belief component
        sigma_belief=0.025,       # std of idiosyncratic belief noise

        # trading rule
        trade_sensitivity=2.4,   # overall sensitivity of trading response to signal

        # price dynamics
        price_impact=0.02,        # impact of aggregate order flow on price
        sigma_price=0.10,         # price noise std
        price_floor=1e-6,         # lower bound to keep price positive

        # risk proxy
        lambda_risk=0.95,         # EWMA memory for risk
        risk_weight=0.05,         # new information weight in risk update

        # initial conditions
        init_price=100.0,
        init_x_std=0.1,
        init_b_std=0.25,

        **kwargs
    ):
        self.P_init = np.array(P_init, dtype=float)
        self.N = self.P_init.shape[0]

        self.seed = seed
        self.horizon = horizon

        self.beta = beta
        self.gamma = gamma
        self.eps_softmax = eps_softmax

        self.rho_y = rho_y
        self.sigma_y = sigma_y

        self.sigma_signal = sigma_signal

        self.alpha_social = alpha_social
        self.sigma_belief = sigma_belief

        self.trade_sensitivity= trade_sensitivity

        self.price_impact = price_impact
        self.sigma_price = sigma_price
        self.price_floor = price_floor

        self.lambda_risk = lambda_risk
        self.risk_weight = risk_weight

        self.init_price = init_price
        self.init_x_std = init_x_std
        self.init_b_std = init_b_std

        # keep compatibility with any external extra parameters
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---------------------------------------------------------
    def reset(self):
        self.t = 0
        self.P = self.P_init.copy()

        self.p = float(self.init_price)
        self.p_prev = float(self.init_price)

        self.x = self.rng.normal(0.0, self.init_x_std, size=self.N)
        self.y = float(self.rng.normal(0.0, self.sigma_y))
        self.b = self.rng.normal(0.0, self.init_b_std, size=self.N)

        self.risk_v = 0.0

        # reputation starts equal across agents
        self.R = np.zeros(self.N)

    # ---------------------------------------------------------
    def _private_signals(self):
        """
        Private signal:
            s_i,t = y_t + epsilon_i,t
        """
        return self.y + self.rng.normal(0.0, self.sigma_signal, size=self.N)

    # ---------------------------------------------------------
    def compute_dynamic_weights(self):
        """
        Compute dynamic influence weights W_t from reputation using:
        1) local neighborhood extraction
        2) local standardization of reputation
        3) softmax mapping

        For each agent i:
            w_ij,t ∝ exp(beta * z_ij,t)
        where z_ij,t is the standardized reputation of neighbor j
        relative to i's neighborhood.
        """
        W = np.zeros_like(self.P)

        for i in range(self.N):
            neighbors = np.where(self.P[i] > 0)[0]

            if len(neighbors) == 0:
                continue

            Rn = self.R[neighbors]
            mean_R = np.mean(Rn)
            std_R = np.std(Rn)

            z = (Rn - mean_R) / (std_R + self.eps_softmax)

            scaled = self.beta * z
            scaled = scaled - np.max(scaled)   # numerical stabilization
            logits = np.exp(scaled)
            weights = logits / np.sum(logits)

            W[i, neighbors] = weights

        return W

    # ---------------------------------------------------------
    def step(self):
        """
        One simulation step.

        Sequence:
        1) latent market state evolves
        2) agents receive private signals
        3) dynamic weights are computed from reputation
        4) beliefs are updated from private + social information
        5) trades are chosen from network-amplified signal
        6) price updates from aggregate order flow
        7) profits are computed using previous positions (x_old)
        8) reputation updates from profits
        9) risk proxy updates from squared returns
        """

        # -----------------------------------------------------
        # 1) latent state process: y_t = rho_y * y_{t-1} + eps_t
        eps_y = float(self.rng.normal(0.0, self.sigma_y))
        self.y = self.rho_y * self.y + eps_y

        # -----------------------------------------------------
        # 2) private signals
        s = self._private_signals()

        # -----------------------------------------------------
        # 3) dynamic influence weights from reputation
        W = self.compute_dynamic_weights()

        # social component uses previous beliefs
        Pb_old = W @ self.b

        # -----------------------------------------------------
        # 4) belief update
        eta = self.rng.normal(0.0, self.sigma_belief, size=self.N)
        self.b = (1.0 - self.alpha_social) * s + self.alpha_social * Pb_old + eta

        # -----------------------------------------------------
        # 5) trading signal
        # Network enters again here by design to amplify social effects.
        signal = W @ self.b

        trade_input = self.trade_sensitivity * signal
        a_fin = np.tanh(trade_input)

        # store old positions for correct profit timing
        x_old = self.x.copy()

        # position update
        delta_x = a_fin
        self.x = self.x + delta_x

        # -----------------------------------------------------
        # 6) price update from aggregate order flow
        net_flow = np.sum(delta_x)
        eps_p = float(self.rng.normal(0.0, self.sigma_price))

        self.p_prev = self.p
        self.p = max(self.p + self.price_impact * net_flow + eps_p, self.price_floor)

        R_price = (self.p - self.p_prev) / self.p_prev

        # -----------------------------------------------------
        # 7) profit update using OLD position
        pi = x_old * R_price

        # -----------------------------------------------------
        # 8) reputation update
        self.R = self.gamma * self.R + (1.0 - self.gamma) * pi

        # -----------------------------------------------------
        # 9) risk update
        self.risk_v = self.lambda_risk * self.risk_v + self.risk_weight * (R_price ** 2)

        row = {
            "t": self.t,
            "price": float(self.p),
            "return": float(R_price),
            "risk_v": float(self.risk_v),
            "belief_var": float(np.var(self.b)),
            "avg_reputation": float(np.mean(self.R)),
            "avg_abs_position": float(np.mean(np.abs(self.x))),
            "net_flow": float(net_flow),
        }

        self.t += 1
        done = self.t >= self.horizon

        return row, done
