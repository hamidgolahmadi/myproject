"""
Fixed-network baseline environment.

This module implements the baseline market model in which the influence
network remains fixed throughout a simulation run.

The environment uses the common mechanisms defined in ``market_core.py`` for:

    - latent-state evolution,
    - private-signal generation,
    - belief updating,
    - network-amplified trading signals,
    - bounded trading,
    - price formation,
    - and EWMA risk updating.

The purpose of keeping this environment separate from the shared core is to
make the experimental distinction explicit:

    Baseline:
        the topology and influence weights remain fixed.

    Adaptive model:
        influence weights may evolve endogenously through reputation.

This file therefore contains only the state management and experiment-specific
logic required for the fixed-network benchmark.
"""

from __future__ import annotations

import numpy as np

from src.metrics.network_metrics import gini_coefficient

from src.model.market_core import (
    compute_social_trading_signal,
    compute_trades,
    generate_private_signals,
    update_beliefs,
    update_latent_state,
    update_price,
    update_risk_ewma,
)


# =============================================================================
# Influence-Matrix Validation
# =============================================================================

def validate_influence_matrix(
    P: np.ndarray,
    tolerance: float = 1e-6,
) -> None:
    """
    Validate the fixed influence matrix used by the baseline environment.

    The model expects a square, non-negative, row-stochastic matrix.

    Parameters
    ----------
    P : np.ndarray
        Candidate influence matrix.

    tolerance : float
        Numerical tolerance used when checking row sums.

    Raises
    ------
    ValueError
        If the matrix is not square, contains negative values, or has rows
        that do not sum approximately to one.
    """

    P = np.asarray(
        P,
        dtype=float,
    )

    # The influence matrix must be two-dimensional and square.
    if (
        P.ndim != 2
        or P.shape[0] != P.shape[1]
    ):
        raise ValueError(
            "Influence matrix P must be square."
        )

    # Negative influence weights are not permitted in the current model.
    if np.any(P < 0.0):
        raise ValueError(
            "Influence matrix P contains negative weights."
        )

    # Every row represents the distribution of influence received by one
    # agent and should therefore sum to one.
    row_sums = P.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Rows of influence matrix P must sum to one."
        )


# =============================================================================
# Fixed-Network Baseline Environment
# =============================================================================

class FixedNetworkMarketEnv:
    """
    Information-network market environment with fixed influence weights.

    The network matrix is copied at reset and remains unchanged throughout
    the simulation. This makes the environment the natural benchmark for
    identifying the effect of topology before introducing endogenous
    reputation-based adaptation.

    Parameters
    ----------
    P_init : np.ndarray
        Initial row-stochastic influence matrix.

    seed : int
        Random seed controlling all stochastic components of the simulation.

    horizon : int
        Maximum number of simulation steps.

    initial_price : float
        Initial market price.

    price_impact : float
        Sensitivity of price changes to aggregate order flow.

    sigma_price : float
        Standard deviation of exogenous price noise.

    price_floor : float
        Minimum permitted market price.

    position_scale : float
        Maximum scale of one-period position changes.

    transaction_cost_rate : float
        Linear transaction-cost coefficient.

    rho_y : float
        Persistence of the latent AR(1) market state.

    sigma_y : float
        Standard deviation of innovations to the latent market state.

    sigma_signal : float
        Standard deviation of private-signal noise.

    alpha_social : float
        Weight placed on social information in belief formation.

    sigma_belief : float
        Standard deviation of idiosyncratic belief noise.

    trade_sensitivity : float
        Sensitivity of bounded trading decisions to the network signal.

    gate_value : float
        Fixed multiplicative gate applied to the trading signal.

    risk_memory : float or None
        Persistence parameter of the EWMA risk proxy. When None, the original
        baseline default of ``2 ** (-1 / 20)`` is used.

    risk_unit : float
        Scaling constant used when reporting the historical ``riskS`` metric.

    init_position_std : float
        Standard deviation of initial positions.

    init_belief_std : float
        Standard deviation of initial beliefs.
    """

    def __init__(
        self,
        P_init: np.ndarray,
        seed: int = 0,
        horizon: int = 3000,
        initial_price: float = 100.0,
        price_impact: float = 0.02,
        sigma_price: float = 0.10,
        price_floor: float = 1e-6,
        position_scale: float = 1.0,
        transaction_cost_rate: float = 0.001,
        rho_y: float = 0.985,
        sigma_y: float = 0.025,
        sigma_signal: float = 0.06,
        alpha_social: float = 0.75,
        sigma_belief: float = 0.025,
        trade_sensitivity: float = 6.0,
        gate_value: float = 0.4,
        risk_memory: float | None = None,
        risk_unit: float = 1e-6,
        init_position_std: float = 0.1,
        init_belief_std: float = 0.25,
    ) -> None:

        # ---------------------------------------------------------------------
        # Network
        # ---------------------------------------------------------------------

        self.P_init = np.asarray(
            P_init,
            dtype=float,
        ).copy()

        validate_influence_matrix(
            self.P_init
        )

        self.N = self.P_init.shape[0]

        # ---------------------------------------------------------------------
        # Simulation controls
        # ---------------------------------------------------------------------

        self.seed = int(seed)
        self.horizon = int(horizon)

        # ---------------------------------------------------------------------
        # Price parameters
        # ---------------------------------------------------------------------

        self.initial_price = float(
            initial_price
        )

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
        # Trading parameters
        # ---------------------------------------------------------------------

        self.position_scale = float(
            position_scale
        )

        self.transaction_cost_rate = float(
            transaction_cost_rate
        )

        self.trade_sensitivity = float(
            trade_sensitivity
        )

        self.gate_value = float(
            gate_value
        )

        # ---------------------------------------------------------------------
        # Latent-state and information parameters
        # ---------------------------------------------------------------------

        self.rho_y = float(
            rho_y
        )

        self.sigma_y = float(
            sigma_y
        )

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
        # Risk parameters
        # ---------------------------------------------------------------------

        # Preserve the original baseline default exactly when no explicit
        # persistence parameter is supplied.
        if risk_memory is None:
            risk_memory = 2 ** (-1 / 20)

        self.risk_memory = float(
            risk_memory
        )

        self.risk_unit = float(
            risk_unit
        )

        # ---------------------------------------------------------------------
        # Initial-condition parameters
        # ---------------------------------------------------------------------

        self.init_position_std = float(
            init_position_std
        )

        self.init_belief_std = float(
            init_belief_std
        )

        # One generator controls the complete stochastic path, making each
        # simulation reproducible from its seed.
        self.rng = np.random.default_rng(
            self.seed
        )

        self.reset()


    # =========================================================================
    # Environment Reset
    # =========================================================================

    def reset(self) -> None:
        """
        Reset the environment to its initial state.

        The influence matrix is restored to its original fixed value.
        """

        self.t = 0

        # The baseline network remains fixed for the complete simulation.
        self.P = self.P_init.copy()

        # ---------------------------------------------------------------------
        # Price state
        # ---------------------------------------------------------------------

        self.p = float(
            self.initial_price
        )

        self.p_prev = float(
            self.initial_price
        )

        self.return_prev = 0.0

        # ---------------------------------------------------------------------
        # Agent positions and cash
        # ---------------------------------------------------------------------

        self.x = self.rng.normal(
            loc=0.0,
            scale=self.init_position_std,
            size=self.N,
        )

        self.c = np.zeros(
            self.N,
            dtype=float,
        )

        self.delta_x_prev = np.zeros(
            self.N,
            dtype=float,
        )

        # ---------------------------------------------------------------------
        # Latent state and information
        # ---------------------------------------------------------------------

        self.y = float(
            self.rng.normal(
                loc=0.0,
                scale=self.sigma_y,
            )
        )

        self.s = generate_private_signals(
            y=self.y,
            n_agents=self.N,
            sigma_signal=self.sigma_signal,
            rng=self.rng,
        )

        self.b = self.rng.normal(
            loc=0.0,
            scale=self.init_belief_std,
            size=self.N,
        )

        # ---------------------------------------------------------------------
        # Risk state
        # ---------------------------------------------------------------------

        self.risk_v = 0.0


    # =========================================================================
    # One Simulation Step
    # =========================================================================

    def step(
        self,
    ) -> tuple[dict, bool]:
        """
        Advance the fixed-network market by one period.

        Sequence
        --------
        1. Evolve the latent market state.
        2. Generate private signals.
        3. Update beliefs using the fixed network.
        4. Construct the network-amplified trading signal.
        5. Convert signals into bounded trades.
        6. Update positions and transaction costs.
        7. Aggregate order flow.
        8. Update market price and return.
        9. Update the EWMA risk proxy.
        10. Return period-level diagnostics.

        Returns
        -------
        tuple[dict, bool]
            Period diagnostics and a flag indicating whether the horizon has
            been reached.
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
        # 3. Belief formation
        # ---------------------------------------------------------------------

        self.b = update_beliefs(
            private_signals=self.s,
            previous_beliefs=self.b,
            influence_matrix=self.P,
            alpha_social=self.alpha_social,
            sigma_belief=self.sigma_belief,
            rng=self.rng,
        )

        # ---------------------------------------------------------------------
        # 4. Network-amplified trading signal
        # ---------------------------------------------------------------------

        signal = compute_social_trading_signal(
            influence_matrix=self.P,
            beliefs=self.b,
        )

        # ---------------------------------------------------------------------
        # 5. Bounded financial action
        # ---------------------------------------------------------------------

        delta_x = compute_trades(
            signal=signal,
            trade_sensitivity=self.trade_sensitivity,
            position_scale=self.position_scale,
            gate=self.gate_value,
        )

        # ---------------------------------------------------------------------
        # 6. Position and transaction-cost update
        # ---------------------------------------------------------------------

        transaction_cost = (
            self.transaction_cost_rate
            * np.abs(delta_x)
        )

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

        self.return_prev = float(
            realised_return
        )

        # ---------------------------------------------------------------------
        # 9. EWMA risk proxy
        # ---------------------------------------------------------------------

        self.risk_v = update_risk_ewma(
            previous_risk=self.risk_v,
            realised_return=realised_return,
            memory=self.risk_memory,
        )

        # ---------------------------------------------------------------------
        # 10. Cash accounting
        # ---------------------------------------------------------------------

        # Preserve the accounting convention used in the original baseline:
        # agents pay the current price for the position change and also incur
        # linear transaction costs.
        self.c = (
            self.c
            - self.p * delta_x
            - transaction_cost
        )

        self.delta_x_prev = (
            delta_x.copy()
        )

        # ---------------------------------------------------------------------
        # 11. Diagnostics
        # ---------------------------------------------------------------------

        # Column sums capture total incoming weighted influence.
        incoming_influence = (
            self.P.sum(axis=0)
        )

        info = {
            "t": int(self.t),

            # Market outcomes
            "price": float(self.p),
            "return": float(realised_return),
            "abs_return": float(
                abs(realised_return)
            ),

            # Risk
            "riskS": float(
                self.risk_v
                / max(
                    self.risk_unit,
                    1e-18,
                )
            ),

            # Order flow
            "net_flow": float(net_flow),
            "flow2": float(
                np.mean(
                    delta_x ** 2
                )
            ),
            "flow_std_cs": float(
                np.std(delta_x)
            ),

            # Belief dynamics
            "belief_var": float(
                np.var(self.b)
            ),
            "belief_range": float(
                np.max(self.b)
                - np.min(self.b)
            ),

            # Network-amplified signal
            "signal_var": float(
                np.var(signal)
            ),
            "mean_abs_signal": float(
                np.mean(
                    np.abs(signal)
                )
            ),
            "mean_abs_deltax": float(
                np.mean(
                    np.abs(delta_x)
                )
            ),

            # Positions
            "position_var": float(
                np.var(self.x)
            ),

            # Fixed-network structural diagnostic
            "gini_in": float(
                gini_coefficient(
                    incoming_influence
                )
            ),

            # The baseline topology does not evolve.
            "dP": 0.0,
        }

        # Advance the simulation clock only after the period is fully complete.
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

# Existing scripts may still import the historical class name. Keeping this
# alias allows those scripts to continue working while the project is migrated
# gradually to the new architecture.
InfoNetworkBaselineEnv = FixedNetworkMarketEnv
