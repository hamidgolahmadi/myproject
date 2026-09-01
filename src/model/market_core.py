"""
Shared core functions for the information-network market model.

This module contains the basic mechanisms that are common to both the
fixed-network baseline model and the adaptive-credibility model.

The functions here deliberately avoid experiment-specific logic. In
particular, this module does not decide whether the influence matrix is fixed
or dynamically updated. That responsibility belongs to the environment using
these functions.

Common model sequence
---------------------
1. Latent market state evolves.
2. Agents receive noisy private signals.
3. Beliefs combine private and social information.
4. Network-weighted beliefs generate trading decisions.
5. Trades create aggregate order flow.
6. Order flow moves the market price.
7. Returns update the risk proxy.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Latent Market State
# =============================================================================

def update_latent_state(
    y_previous: float,
    rho_y: float,
    sigma_y: float,
    rng: np.random.Generator,
) -> float:
    """
    Evolve the latent market state using an AR(1) process.

    The state follows

        y_t = rho_y * y_{t-1} + epsilon_t,

    where epsilon_t is normally distributed with standard deviation sigma_y.

    Parameters
    ----------
    y_previous : float
        Previous latent market state.

    rho_y : float
        Persistence parameter of the AR(1) process.

    sigma_y : float
        Standard deviation of the latent-state innovation.

    rng : np.random.Generator
        Random-number generator used for reproducibility.

    Returns
    -------
    float
        Updated latent market state.
    """

    innovation = float(
        rng.normal(
            loc=0.0,
            scale=sigma_y,
        )
    )

    return float(
        rho_y * y_previous
        + innovation
    )


# =============================================================================
# Private Information
# =============================================================================

def generate_private_signals(
    y: float,
    n_agents: int,
    sigma_signal: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate noisy private signals around the latent market state.

    For each agent i,

        s_i,t = y_t + epsilon_i,t.

    Parameters
    ----------
    y : float
        Current latent market state.

    n_agents : int
        Number of agents.

    sigma_signal : float
        Standard deviation of idiosyncratic signal noise.

    rng : np.random.Generator
        Random-number generator.

    Returns
    -------
    np.ndarray
        Vector of private signals.
    """

    noise = rng.normal(
        loc=0.0,
        scale=sigma_signal,
        size=n_agents,
    )

    return (
        float(y)
        + noise
    )


# =============================================================================
# Belief Formation
# =============================================================================

def update_beliefs(
    private_signals: np.ndarray,
    previous_beliefs: np.ndarray,
    influence_matrix: np.ndarray,
    alpha_social: float,
    sigma_belief: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Update beliefs using private information and social influence.

    The belief rule is

        b_t =
            (1 - alpha_social) * s_t
            + alpha_social * W_t b_{t-1}
            + eta_t.

    The same function can therefore be used with either a fixed influence
    matrix or a dynamically updated matrix.

    Parameters
    ----------
    private_signals : np.ndarray
        Current private signals.

    previous_beliefs : np.ndarray
        Beliefs from the previous period.

    influence_matrix : np.ndarray
        Current row-stochastic influence matrix.

    alpha_social : float
        Weight placed on the social component.

    sigma_belief : float
        Standard deviation of idiosyncratic belief noise.

    rng : np.random.Generator
        Random-number generator.

    Returns
    -------
    np.ndarray
        Updated belief vector.
    """

    private_signals = np.asarray(
        private_signals,
        dtype=float,
    )

    previous_beliefs = np.asarray(
        previous_beliefs,
        dtype=float,
    )

    influence_matrix = np.asarray(
        influence_matrix,
        dtype=float,
    )

    # Social information is based on neighbours' previous-period beliefs.
    social_component = (
        influence_matrix
        @ previous_beliefs
    )

    # Idiosyncratic noise prevents perfectly deterministic belief evolution.
    belief_noise = rng.normal(
        loc=0.0,
        scale=sigma_belief,
        size=previous_beliefs.size,
    )

    beliefs = (
        (1.0 - alpha_social) * private_signals
        + alpha_social * social_component
        + belief_noise
    )

    return beliefs


# =============================================================================
# Network-Amplified Trading Signal
# =============================================================================

def compute_social_trading_signal(
    influence_matrix: np.ndarray,
    beliefs: np.ndarray,
) -> np.ndarray:
    """
    Construct the network-weighted signal used for trading.

    Parameters
    ----------
    influence_matrix : np.ndarray
        Current influence matrix.

    beliefs : np.ndarray
        Current agent beliefs.

    Returns
    -------
    np.ndarray
        Network-aggregated trading signal.
    """

    return (
        np.asarray(
            influence_matrix,
            dtype=float,
        )
        @ np.asarray(
            beliefs,
            dtype=float,
        )
    )


# =============================================================================
# Bounded Trading Rule
# =============================================================================

def compute_trades(
    signal: np.ndarray,
    trade_sensitivity: float,
    position_scale: float = 1.0,
    gate: float | np.ndarray = 1.0,
) -> np.ndarray:
    """
    Convert trading signals into bounded position changes.

    The trading rule is

        delta_x_i =
            position_scale
            * tanh(
                trade_sensitivity
                * gate_i
                * signal_i
            ).

    This general form covers both existing implementations:

    - the baseline model can use its fixed gate and x_max;
    - the adaptive model can use gate = 1 and position_scale = 1.

    Parameters
    ----------
    signal : np.ndarray
        Network-amplified trading signal.

    trade_sensitivity : float
        Sensitivity of trading behaviour to the signal.

    position_scale : float, optional
        Maximum absolute size of a one-period position change.
        Default is 1.

    gate : float or np.ndarray, optional
        Multiplicative trading gate. Default is 1.

    Returns
    -------
    np.ndarray
        Bounded position changes.
    """

    signal = np.asarray(
        signal,
        dtype=float,
    )

    trade_input = (
        float(trade_sensitivity)
        * np.asarray(gate, dtype=float)
        * signal
    )

    return (
        float(position_scale)
        * np.tanh(trade_input)
    )


# =============================================================================
# Price Dynamics
# =============================================================================

def update_price(
    previous_price: float,
    net_flow: float,
    price_impact: float,
    sigma_price: float,
    price_floor: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Update market price from aggregate order flow.

    Price evolves according to

        p_t =
            p_{t-1}
            + price_impact * F_t
            + epsilon_t,

    subject to a positive lower bound.

    The corresponding simple return is

        R_t = (p_t - p_{t-1}) / p_{t-1}.

    Parameters
    ----------
    previous_price : float
        Price before the current update.

    net_flow : float
        Aggregate order flow.

    price_impact : float
        Sensitivity of price to aggregate order flow.

    sigma_price : float
        Standard deviation of exogenous price noise.

    price_floor : float
        Minimum permitted price.

    rng : np.random.Generator
        Random-number generator.

    Returns
    -------
    tuple[float, float]
        Updated price and realised simple return.
    """

    price_noise = float(
        rng.normal(
            loc=0.0,
            scale=sigma_price,
        )
    )

    new_price = max(
        float(previous_price)
        + float(price_impact) * float(net_flow)
        + price_noise,
        float(price_floor),
    )

    realised_return = (
        new_price
        - float(previous_price)
    ) / float(previous_price)

    return (
        float(new_price),
        float(realised_return),
    )


# =============================================================================
# Risk Updating
# =============================================================================

def update_risk_ewma(
    previous_risk: float,
    realised_return: float,
    memory: float,
    innovation_weight: float | None = None,
) -> float:
    """
    Update an EWMA-style return-risk proxy.

    If innovation_weight is omitted, the conventional EWMA specification is

        risk_t =
            memory * risk_{t-1}
            + (1 - memory) * R_t^2.

    An explicit innovation weight can also be supplied when reproducing model
    specifications in which the two coefficients are stored separately.

    Parameters
    ----------
    previous_risk : float
        Previous value of the risk proxy.

    realised_return : float
        Current market return.

    memory : float
        Persistence placed on previous risk.

    innovation_weight : float or None, optional
        Weight placed on the current squared return. When None, it is set to
        1 - memory.

    Returns
    -------
    float
        Updated risk proxy.
    """

    if innovation_weight is None:
        innovation_weight = (
            1.0
            - float(memory)
        )

    return float(
        float(memory) * float(previous_risk)
        + float(innovation_weight)
        * float(realised_return) ** 2
    )


# =============================================================================
# Profit Calculation
# =============================================================================

def compute_profits(
    previous_positions: np.ndarray,
    realised_return: float,
) -> np.ndarray:
    """
    Compute agent profits using positions held before the price move.

    This timing convention is important: current trades should not earn the
    return that is realised within the same step.

    Parameters
    ----------
    previous_positions : np.ndarray
        Positions held before the current trading and price update.

    realised_return : float
        Current realised market return.

    Returns
    -------
    np.ndarray
        Agent-level realised profits.
    """

    previous_positions = np.asarray(
        previous_positions,
        dtype=float,
    )

    return (
        previous_positions
        * float(realised_return)
    )


# =============================================================================
# Reputation Updating
# =============================================================================

def update_reputation(
    previous_reputation: np.ndarray,
    profits: np.ndarray,
    memory: float,
) -> np.ndarray:
    """
    Update agent reputation using an exponentially weighted moving average.

    The reputation rule is

        R_i,t =
            memory * R_i,t-1
            + (1 - memory) * pi_i,t.

    Parameters
    ----------
    previous_reputation : np.ndarray
        Previous agent reputations.

    profits : np.ndarray
        Current realised profits.

    memory : float
        Persistence parameter for reputation.

    Returns
    -------
    np.ndarray
        Updated reputation vector.
    """

    previous_reputation = np.asarray(
        previous_reputation,
        dtype=float,
    )

    profits = np.asarray(
        profits,
        dtype=float,
    )

    return (
        float(memory) * previous_reputation
        + (1.0 - float(memory)) * profits
    )
