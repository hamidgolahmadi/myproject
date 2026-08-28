"""
Adaptive-credibility information-network market environment.

This module implements the endogenous influence-weight version of the market
model. The underlying network support remains fixed, but the influence weights
assigned to existing neighbours evolve over time according to agent reputation.

The model follows the sequence:

    1. The latent market state evolves.
    2. Agents receive noisy private signals.
    3. Reputation determines dynamic influence weights.
    4. Beliefs combine private and social information.
    5. Network-weighted beliefs generate bounded trades.
    6. Aggregate order flow moves the market price.
    7. Profits are calculated using positions held before the price move.
    8. Reputation is updated from realised profits.
    9. An EWMA-style risk proxy is updated from squared returns.

The common economic mechanisms are imported from ``market_core.py`` so that
the fixed-network baseline and adaptive model use exactly the same underlying
market dynamics wherever their specifications overlap.
"""

from __future__ import annotations

import numpy as np

from src.model.market_core import (
    compute_profits,
    compute_social_trading_signal,
    compute_trades,
    generate_private_signals,
    update_beliefs,
    update_latent_state,
    update_price,
    update_reputation,
    update_risk_ewma,
)


# =============================================================================
# Adaptive-Credibility Environment
# =============================================================================

class AdaptiveCredibilityMarketEnv:
    """
    Market environment with reputation-based endogenous influence weights.

    The binary support of the information network is fixed by ``P_init``.
    However, the weights placed on existing neighbours evolve endogenously
    according to relative reputation.

    For each agent i, neighbour reputations are standardised locally:

        z_ij,t =
            (R_j,t - mean(R_Ni,t))
            / (std(R_Ni,t) + eps)

    Influence weights are then obtained through a softmax rule:

        w_ij,t ∝ exp(beta * z_ij,t).

    Parameters
    ----------
    P_init : np.ndarray
        Initial influence matrix. Positive entries determine which neighbour
        relationships are available. The initial magnitudes are not used as
        adaptive weights once the simulation begins; the network support is
        retained while weights are recomputed from reputation.

    seed : int
        Random seed controlling the complete stochastic simulation path.

    horizon : int
        Number of periods in the simulation.

    beta : float
        Sensitivity of influence weights to relative reputation. Higher beta
        produces stronger concentration on locally high-reputation neighbours.

    gamma : float
        Persistence parameter in the reputation EWMA.

    eps_softmax : float
        Small numerical constant used when standardising reputation.

    rho_y : float
        Persistence of the latent AR(1) market state.

    sigma_y : float
        Standard deviation of latent-state innovations.

    sigma_signal : float
        Standard deviation of private-signal noise.

    alpha_social : float
        Weight placed on social information in belief formation.

    sigma_belief : float
        Standard deviation of idiosyncratic belief noise.

    trade_sensitivity : float
        Sensitivity of bounded trading decisions to the network signal.

    price_impact : float
        Sensitivity of price changes to aggregate order flow.

    sigma_price : float
        Standard deviation of exogenous price noise.

    price_floor : float
        Minimum permitted market price.

    lambda_risk : float
        Persistence parameter in the risk EWMA.

    risk_weight : float
        Weight placed on the current squared return in the risk update.

    init_price : float
        Initial market price.

    init_x_std : float
        Standard deviation of initial agent positions.

    init_b_std : float
        Standard deviation of initial beliefs.

    **kwargs
        Additional parameters are retained as attributes for backward
        compatibility with older experiment scripts.
    """

    def __init__(
        self,
        P_init: np.ndarray,
        seed: int = 0,
        horizon: int = 3000,
        beta: float = 1.0,
        gamma: float = 0.9,
        eps_softmax: float = 1e-8,
        rho_y: float = 0.985,
        sigma_y: float = 0.025,
        sigma_signal: float = 0.06,
        alpha_social: float = 0.75,
        sigma_belief: float = 0.025,
        trade_sensitivity: float = 2.4,
        price_impact: float = 0.02,
        sigma_price: float = 0.10,
        price_floor: float = 1e-6,
        lambda_risk: float = 0.95,
        risk_weight: float = 0.05,
        init_price: float = 100.0,
        init_x_std: float = 0.1,
        init_b_std: float = 0.25,
        **kwargs,
    ) -> None:

        # ---------------------------------------------------------------------
        # Network structure
        # ---------------------------------------------------------------------

        self.P_init = np.asarray(
            P_init,
            dtype=float,
        ).copy()

        if (
            self.P_init.ndim != 2
            or self.P_init.shape[0] != self.P_init.shape[1]
        ):
            raise ValueError(
                "P_init must be a square influence matrix."
            )

        if np.any(self.P_init < 0.0):
            raise ValueError(
                "P_init cannot contain negative influence weights."
            )

        self.N = self.P_init.shape[0]

        # ---------------------------------------------------------------------
        # Simulation controls
        # ---------------------------------------------------------------------

        self.seed = int(seed)
        self.horizon = int(horizon)

        # ---------------------------------------------------------------------
        # Reputation-based weighting parameters
        # ---------------------------------------------------------------------

        # beta controls how sharply reputation differences translate into
        # differences in influence.
        self.beta = float(beta)

        # gamma controls persistence in the reputation process.
        self.gamma = float(gamma)

        # Numerical safeguard for local reputation standardisation.
        self.eps_softmax = float(
            eps_softmax
        )

        # ---------------------------------------------------------------------
        # Latent-state parameters
        # ---------------------------------------------------------------------

        self.rho_y = float(rho_y)
        self.sigma_y = float(sigma_y)

        # ---------------------------------------------------------------------
        # Information and belief parameters
        # ---------------------------------------------------------------------

        self.sigma_signal = float(
            sigma_signal
        )

        self.alpha_social = float(
            alpha_social
        )

        self.sigma_belief = float(
            sigma_belief
        )

        # ---------------------------------------------------------------------
        # Trading parameters
        # ---------------------------------------------------------------------

        self.trade_sensitivity = float(
            trade_sensitivity
        )

        # ---------------------------------------------------------------------
        # Price parameters
        # ---------------------------------------------------------------------

        self.price_impact = float(
            price_impact
        )

        self.sigma_price = float(
            sigma_price
        )

        self.price_floor = float(
            price_floor
        )

        # ---------------------------------------------------------------------
        # Risk parameters
        # ---------------------------------------------------------------------

        self.lambda_risk = float(
            lambda_risk
        )

        self.risk_weight = float(
            risk_weight
        )

        # ---------------------------------------------------------------------
        # Initial conditions
        # ---------------------------------------------------------------------

        self.init_price = float(
            init_price
        )

        self.init_x_std = float(
            init_x_std
        )

        self.init_b_std = float(
            init_b_std
        )

        # ---------------------------------------------------------------------
        # Backward compatibility
        # ---------------------------------------------------------------------

        # Some older experiment scripts may pass additional configuration
        # variables. Retaining them as attributes allows gradual migration
        # without silently discarding those parameters.
        for key, value in kwargs.items():
            setattr(
                self,
                key,
                value,
            )

        # One random-number generator controls the complete stochastic path.
        self.rng = np.random.default_rng(
            self.seed
        )

        self.reset()


    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """
        Reset all market and agent states.

        Reputation starts equally across agents. Consequently, the first
        adaptive weight matrix assigns equal influence to all available
        neighbours within each row.
        """

        self.t = 0

        # P stores the fixed support of the network.
        # Positive entries indicate which influence relationships are allowed.
        self.P = self.P_init.copy()

        # The current adaptive matrix will be created from reputation.
        self.W = self.P_init.copy()

        # ---------------------------------------------------------------------
        # Price state
        # ---------------------------------------------------------------------

        self.p = float(
            self.init_price
        )

        self.p_prev = float(
            self.init_price
        )

        # ---------------------------------------------------------------------
        # Agent positions
        # ---------------------------------------------------------------------

        self.x = self.rng.normal(
            loc=0.0,
            scale=self.init_x_std,
            size=self.N,
        )

        # ---------------------------------------------------------------------
        # Latent market state
        # ---------------------------------------------------------------------

        self.y = float(
            self.rng.normal(
                loc=0.0,
                scale=self.sigma_y,
            )
        )

        # ---------------------------------------------------------------------
        # Beliefs
        # ---------------------------------------------------------------------

        self.b = self.rng.normal(
            loc=0.0,
            scale=self.init_b_std,
            size=self.N,
        )

        # ---------------------------------------------------------------------
        # Risk
        # ---------------------------------------------------------------------

        self.risk_v = 0.0

        # ---------------------------------------------------------------------
        # Reputation
        # ---------------------------------------------------------------------

        # Equal initial reputation ensures that the initial endogenous weight
        # distribution is neutral within each agent's neighbourhood.
        self.R = np.zeros(
            self.N,
            dtype=float,
        )


    # =========================================================================
    # Dynamic Influence Weights
    # =========================================================================

    def compute_dynamic_weights(
        self,
    ) -> np.ndarray:
        """
        Construct reputation-based influence weights.

        The topology support is fixed: an agent can only assign positive
        influence to nodes that are already neighbours in ``self.P``.

        Within each neighbourhood:

        1. Neighbour reputations are extracted.
        2. Reputation is standardised relative to that local neighbourhood.
        3. ``beta`` scales the standardised reputation scores.
        4. A numerically stable softmax converts scores into weights.

        Returns
        -------
        np.ndarray
            Row-stochastic adaptive influence matrix.
        """

        W = np.zeros_like(
            self.P,
            dtype=float,
        )

        for i in range(self.N):

            # Identify the set of structurally available neighbours.
            neighbors = np.where(
                self.P[i] > 0.0
            )[0]

            # If an agent has no neighbours, its row remains zero.
            if neighbors.size == 0:
                continue

            # Reputation of agents in i's local information neighbourhood.
            local_reputation = self.R[
                neighbors
            ]

            local_mean = float(
                np.mean(
                    local_reputation
                )
            )

            local_std = float(
                np.std(
                    local_reputation
                )
            )

            # Standardise reputation within the local neighbourhood.
            z = (
                local_reputation
                - local_mean
            ) / (
                local_std
                + self.eps_softmax
            )

            # beta controls reputational selectivity.
            scaled_scores = (
                self.beta
                * z
            )

            # Subtracting the largest score prevents overflow when exponentiating
            # without changing the resulting softmax probabilities.
            scaled_scores = (
                scaled_scores
                - np.max(
                    scaled_scores
                )
            )

            logits = np.exp(
                scaled_scores
            )

            weights = (
                logits
                / np.sum(logits)
            )

            W[
                i,
                neighbors
            ] = weights

        return W


    # =========================================================================
    # One Simulation Step
    # =========================================================================

    def step(
        self,
    ) -> tuple[dict, bool]:
        """
        Advance the adaptive market by one period.

        Sequence
        --------
        1. Evolve the latent market state.
        2. Generate noisy private signals.
        3. Compute reputation-based influence weights.
        4. Update beliefs using private and social information.
        5. Construct the network-amplified trading signal.
        6. Generate bounded position changes.
        7. Update positions and aggregate order flow.
        8. Update market price and return.
        9. Compute profits using positions held before the price move.
        10. Update reputation from realised profits.
        11. Update the return-risk proxy.
        12. Return period-level diagnostics.

        Returns
        -------
        tuple[dict, bool]
            Period diagnostics and a flag indicating whether the simulation
            horizon has been reached.
        """

        # ---------------------------------------------------------------------
        # 1. Latent market state
        # ---------------------------------------------------------------------

        self.y = update_latent_state(
            y_previous=self.y,
            rho_y=self.rho_y,
            sigma_y=self.sigma_y,
            rng=self.rng,
        )

        # ---------------------------------------------------------------------
        # 2. Private signals
        # ---------------------------------------------------------------------

        self.s = generate_private_signals(
            y=self.y,
            n_agents=self.N,
            sigma_signal=self.sigma_signal,
            rng=self.rng,
        )

        # ---------------------------------------------------------------------
        # 3. Reputation-based dynamic influence weights
        # ---------------------------------------------------------------------

        self.W = (
            self.compute_dynamic_weights()
        )

        # ---------------------------------------------------------------------
        # 4. Belief formation
        # ---------------------------------------------------------------------

        # Social information uses previous-period beliefs, consistent with the
        # lagged DeGroot-style timing of the original implementation.
        self.b = update_beliefs(
            private_signals=self.s,
            previous_beliefs=self.b,
            influence_matrix=self.W,
            alpha_social=self.alpha_social,
            sigma_belief=self.sigma_belief,
            rng=self.rng,
        )

        # ---------------------------------------------------------------------
        # 5. Network-amplified trading signal
        # ---------------------------------------------------------------------

        signal = compute_social_trading_signal(
            influence_matrix=self.W,
            beliefs=self.b,
        )

        # ---------------------------------------------------------------------
        # 6. Bounded trading decision
        # ---------------------------------------------------------------------

        # position_scale=1 and gate=1 reproduce the historical adaptive
        # specification exactly:
        #
        #     delta_x = tanh(trade_sensitivity * signal)
        #
        delta_x = compute_trades(
            signal=signal,
            trade_sensitivity=self.trade_sensitivity,
            position_scale=1.0,
            gate=1.0,
        )

        # IMPORTANT:
        # Profit must be based on the position held before this period's trade
        # and price movement. This prevents look-ahead timing.
        x_old = self.x.copy()

        # Update current positions.
        self.x = (
            self.x
            + delta_x
        )

        # ---------------------------------------------------------------------
        # 7. Aggregate order flow
        # ---------------------------------------------------------------------

        net_flow = float(
            np.sum(delta_x)
        )

        # ---------------------------------------------------------------------
        # 8. Price and return
        # ---------------------------------------------------------------------

        self.p_prev = float(
            self.p
        )

        self.p, realised_return = update_price(
            previous_price=self.p_prev,
            net_flow=net_flow,
            price_impact=self.price_impact,
            sigma_price=self.sigma_price,
            price_floor=self.price_floor,
            rng=self.rng,
        )

        # ---------------------------------------------------------------------
        # 9. Realised profits
        # ---------------------------------------------------------------------

        self.pi = compute_profits(
            previous_positions=x_old,
            realised_return=realised_return,
        )

        # ---------------------------------------------------------------------
        # 10. Reputation update
        # ---------------------------------------------------------------------

        self.R = update_reputation(
            previous_reputation=self.R,
            profits=self.pi,
            memory=self.gamma,
        )

        # ---------------------------------------------------------------------
        # 11. Risk update
        # ---------------------------------------------------------------------

        # The historical adaptive specification stores the persistence and
        # innovation coefficients separately, so risk_weight is passed
        # explicitly rather than automatically using 1 - lambda_risk.
        self.risk_v = update_risk_ewma(
            previous_risk=self.risk_v,
            realised_return=realised_return,
            memory=self.lambda_risk,
            innovation_weight=self.risk_weight,
        )

        # ---------------------------------------------------------------------
        # 12. Period diagnostics
        # ---------------------------------------------------------------------

        info = {
            "t": int(self.t),
            "price": float(self.p),
            "return": float(
                realised_return
            ),
            "risk_v": float(
                self.risk_v
            ),
            "belief_var": float(
                np.var(self.b)
            ),
            "avg_reputation": float(
                np.mean(self.R)
            ),
            "avg_abs_position": float(
                np.mean(
                    np.abs(self.x)
                )
            ),
            "net_flow": float(
                net_flow
            ),
        }

        # Advance the simulation clock only after all period calculations have
        # been completed.
        self.t += 1

        done = (
            self.t
            >= self.horizon
        )

        return (
            info,
            done,
        )


# =============================================================================
# Backward-Compatible Alias
# =============================================================================

# Older experiment scripts use the historical class name. Keeping the alias
# allows them to continue working while the project is migrated incrementally.
InfoNetworkAdaptiveEnv = AdaptiveCredibilityMarketEnv
