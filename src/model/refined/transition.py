"""Canonical one-period transition coordinator for Equation (39) and Equations (80)-(82)."""

from __future__ import annotations

import numpy as np

from .attention import standardised_reputation_scores, update_attention
from .beliefs import update_beliefs
from .fundamentals import fundamental_value, private_signals, update_fundamental
from .market import market_return, update_price
from .parameters import RefinedParameters
from .reputation import realised_profits, update_reputation
from .shocks import PeriodShocks
from .state import PeriodOutputs, RefinedState, validate_attention, validate_graph_support
from .trading import (
    desired_actions,
    execute_actions,
    net_order_flow,
    perceived_values,
    update_positions,
    valuation_gaps,
)


def transition_one_period(
    state: RefinedState,
    shocks: PeriodShocks,
    graph: np.ndarray,
    parameters: RefinedParameters,
    *,
    adaptive_attention: bool = True,
) -> tuple[RefinedState, PeriodOutputs]:
    """Advance the refined model by one period using the report's event timing.

    The implemented sequence is the sequential ABM in Equation (39):

    ``theta_t, v_t, s_t -> b_t -> vhat_t -> m_t -> desired action``
    ``-> executed action -> x_t -> F_t -> p_t -> r_t -> pi_t -> R_t``
    ``-> z_t -> W_t``.

    Crucially, current beliefs use inherited ``state.attention = W_{t-1}``.
    The newly computed ``W_t`` enters only the returned state and can therefore
    affect beliefs from the next period onward.  No contemporaneous fixed point
    or matrix inverse is solved.

    ``adaptive_attention=False`` implements the fixed-influence benchmark by
    carrying the inherited valid attention matrix forward unchanged.  Relative
    reputation scores are still returned as diagnostics.
    """

    if not isinstance(state, RefinedState):
        raise TypeError("state must be a RefinedState")
    if not isinstance(shocks, PeriodShocks):
        raise TypeError("shocks must be a PeriodShocks")
    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be a RefinedParameters")
    if not isinstance(adaptive_attention, bool):
        raise TypeError("adaptive_attention must be a bool")

    graph_array = validate_graph_support(graph)
    state.validate_against(graph_array, parameters.x_bar)
    if shocks.n_agents != state.n_agents:
        raise ValueError("shock dimension does not match state dimension")

    # Exogenous information block: Equations (42), (44)-(46), and (80).
    theta = update_fundamental(state.theta, shocks.u_theta, parameters)
    value = fundamental_value(theta, parameters)
    signals = private_signals(theta, shocks.epsilon_s)

    # Current beliefs use W_{t-1}, never W_t: Equations (48)-(50).
    beliefs = update_beliefs(
        signals,
        state.beliefs,
        state.attention,
        shocks.epsilon_b,
        parameters.alpha,
    )

    # Trading and inventory block: Equations (63)-(72) and (81).
    perceived = perceived_values(
        beliefs,
        v_bar=parameters.v_bar,
        psi=parameters.psi,
    )
    gaps = valuation_gaps(perceived, lagged_price=state.price)
    desired = desired_actions(gaps, kappa=parameters.kappa)
    actions = execute_actions(
        desired,
        state.positions,
        x_bar=parameters.x_bar,
    )
    positions = update_positions(state.positions, actions)
    order_flow = net_order_flow(actions)

    # Price and performance block: Equations (74)-(79).
    price = update_price(
        previous_price=state.price,
        fundamental_value=value,
        net_order_flow=order_flow,
        chi=parameters.chi,
        lambda_price=parameters.lambda_price,
        sigma_p=parameters.sigma_p,
        epsilon_p=shocks.epsilon_p,
    )
    return_ = market_return(previous_price=state.price, current_price=price)
    profits = realised_profits(
        inherited_positions=state.positions,
        return_=return_,
    )
    reputation = update_reputation(
        previous_reputation=state.reputation,
        profits=profits,
        gamma_R=parameters.gamma_R,
    )

    # End-of-period adaptive feedback: Equations (57)-(60) and (82).
    if adaptive_attention:
        attention, scores = update_attention(
            reputation,
            graph_array,
            parameters.beta,
            parameters.sigma_0,
        )
    else:
        attention = validate_attention(state.attention, graph_array)
        scores = standardised_reputation_scores(
            reputation,
            graph_array,
            parameters.sigma_0,
        )

    next_state = RefinedState(
        theta=theta,
        beliefs=beliefs,
        positions=positions,
        price=price,
        reputation=reputation,
        attention=attention,
    )
    next_state.validate_against(graph_array, parameters.x_bar)

    outputs = PeriodOutputs(
        fundamental_value=value,
        signals=signals,
        perceived_values=perceived,
        valuation_gaps=gaps,
        desired_actions=desired,
        actions=actions,
        net_order_flow=order_flow,
        return_=return_,
        profits=profits,
        reputation_scores=scores,
    )
    return next_state, outputs
