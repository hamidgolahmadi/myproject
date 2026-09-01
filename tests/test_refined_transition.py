"""End-to-end tests for the canonical refined one-period transition."""

import numpy as np
import pytest

from src.model.refined import (
    PeriodShocks,
    RefinedParameters,
    RefinedState,
    transition_one_period,
)


def make_parameters(**overrides):
    values = dict(
        rho_theta=0.5,
        sigma_theta=0.1,
        v_bar=1.0,
        psi=2.0,
        sigma_s=0.1,
        sigma_b=0.05,
        alpha=0.4,
        kappa=1.5,
        x_bar=1.0,
        chi=0.25,
        lambda_price=0.1,
        sigma_p=0.2,
        gamma_R=0.5,
        beta=2.0,
        sigma_0=0.5,
    )
    values.update(overrides)
    return RefinedParameters(**values)


def complete_graph_three():
    return np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )


def base_state(*, reputation=None, attention=None):
    if reputation is None:
        reputation = np.array([0.3, -0.1, 0.2])
    if attention is None:
        attention = np.array(
            [
                [0.0, 0.75, 0.25],
                [0.60, 0.0, 0.40],
                [0.20, 0.80, 0.0],
            ]
        )
    return RefinedState(
        theta=0.2,
        beliefs=np.array([0.1, -0.2, 0.3]),
        positions=np.array([0.2, -0.4, 0.1]),
        price=1.1,
        reputation=np.asarray(reputation, dtype=float),
        attention=np.asarray(attention, dtype=float),
    )


def base_shocks():
    return PeriodShocks(
        u_theta=0.05,
        epsilon_s=np.array([0.02, -0.01, 0.03]),
        epsilon_b=np.array([0.01, -0.02, 0.00]),
        epsilon_p=0.5,
    )


def _manual_softmax_attention(reputation, graph, beta, sigma_0):
    n = graph.shape[0]
    scores = np.zeros((n, n), dtype=float)
    attention = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbours = np.flatnonzero(graph[i])
        local = reputation[neighbours]
        mean = np.mean(local)
        dispersion = np.sqrt(np.mean((local - mean) ** 2) + sigma_0**2)
        scores[i, neighbours] = (local - mean) / dispersion
        logits = beta * scores[i, neighbours]
        shifted = logits - np.max(logits)
        weights = np.exp(shifted)
        attention[i, neighbours] = weights / np.sum(weights)

    return attention, scores


def test_transition_matches_manual_equation_39_sequence():
    params = make_parameters()
    graph = complete_graph_three()
    state = base_state()
    shocks = base_shocks()

    next_state, outputs = transition_one_period(state, shocks, graph, params)

    theta = params.rho_theta * state.theta + shocks.u_theta
    value = params.v_bar + params.psi * theta
    signals = theta + shocks.epsilon_s
    beliefs = (
        (1.0 - params.alpha) * signals
        + params.alpha * (state.attention @ state.beliefs)
        + shocks.epsilon_b
    )
    perceived = params.v_bar + params.psi * beliefs
    gaps = perceived - state.price
    desired = np.tanh(params.kappa * gaps)
    lower = -params.x_bar - state.positions
    upper = params.x_bar - state.positions
    actions = np.minimum(upper, np.maximum(lower, desired))
    positions = state.positions + actions
    order_flow = np.sum(actions)
    delta_price = (
        params.chi * (value - state.price)
        + params.lambda_price * order_flow
        + params.sigma_p * shocks.epsilon_p
    )
    price = state.price + delta_price
    return_ = price - state.price
    profits = state.positions * return_
    reputation = params.gamma_R * state.reputation + (1.0 - params.gamma_R) * profits
    attention, scores = _manual_softmax_attention(
        reputation,
        graph,
        params.beta,
        params.sigma_0,
    )

    assert next_state.theta == pytest.approx(theta)
    assert np.allclose(next_state.beliefs, beliefs)
    assert np.allclose(next_state.positions, positions)
    assert next_state.price == pytest.approx(price)
    assert np.allclose(next_state.reputation, reputation)
    assert np.allclose(next_state.attention, attention)

    assert outputs.fundamental_value == pytest.approx(value)
    assert np.allclose(outputs.signals, signals)
    assert np.allclose(outputs.perceived_values, perceived)
    assert np.allclose(outputs.valuation_gaps, gaps)
    assert np.allclose(outputs.desired_actions, desired)
    assert np.allclose(outputs.actions, actions)
    assert outputs.net_order_flow == pytest.approx(order_flow)
    assert outputs.return_ == pytest.approx(return_)
    assert np.allclose(outputs.profits, profits)
    assert np.allclose(outputs.reputation_scores, scores)


def test_current_beliefs_use_lagged_attention_not_end_of_period_attention():
    params = make_parameters(beta=4.0)
    graph = complete_graph_three()
    shocks = base_shocks()

    state_a = base_state(reputation=np.array([8.0, -5.0, 1.0]))
    state_b = base_state(reputation=np.array([-6.0, 7.0, 0.5]))

    next_a, outputs_a = transition_one_period(state_a, shocks, graph, params)
    next_b, outputs_b = transition_one_period(state_b, shocks, graph, params)

    assert np.allclose(next_a.beliefs, next_b.beliefs)
    assert np.allclose(outputs_a.actions, outputs_b.actions)
    assert outputs_a.net_order_flow == pytest.approx(outputs_b.net_order_flow)
    assert next_a.price == pytest.approx(next_b.price)
    assert not np.allclose(next_a.attention, next_b.attention)


def test_realised_profit_uses_inherited_not_updated_positions():
    params = make_parameters()
    graph = complete_graph_three()
    state = base_state()
    shocks = base_shocks()

    next_state, outputs = transition_one_period(state, shocks, graph, params)

    expected = state.positions * outputs.return_
    contemporaneous_wrong = next_state.positions * outputs.return_

    assert np.allclose(outputs.profits, expected)
    assert not np.allclose(outputs.profits, contemporaneous_wrong)


def test_fixed_attention_benchmark_carries_inherited_attention_forward():
    params = make_parameters(beta=9.0)
    graph = complete_graph_three()
    state = base_state()

    next_state, outputs = transition_one_period(
        state,
        base_shocks(),
        graph,
        params,
        adaptive_attention=False,
    )

    assert np.allclose(next_state.attention, state.attention)
    assert np.all(np.isfinite(outputs.reputation_scores))


def test_alpha_zero_removes_graph_from_current_market_block():
    params = make_parameters(alpha=0.0)
    shocks = base_shocks()

    graph_a = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )
    attention_a = graph_a.astype(float)

    graph_b = complete_graph_three()
    attention_b = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )

    state_a = base_state(attention=attention_a)
    state_b = base_state(attention=attention_b)

    next_a, outputs_a = transition_one_period(state_a, shocks, graph_a, params)
    next_b, outputs_b = transition_one_period(state_b, shocks, graph_b, params)

    assert np.allclose(next_a.beliefs, next_b.beliefs)
    assert np.allclose(outputs_a.actions, outputs_b.actions)
    assert outputs_a.net_order_flow == pytest.approx(outputs_b.net_order_flow)
    assert next_a.price == pytest.approx(next_b.price)
    assert np.allclose(outputs_a.profits, outputs_b.profits)
    assert np.allclose(next_a.reputation, next_b.reputation)


def test_transition_rejects_shock_dimension_mismatch():
    params = make_parameters()
    graph = complete_graph_three()
    shocks = PeriodShocks(
        u_theta=0.0,
        epsilon_s=np.zeros(2),
        epsilon_b=np.zeros(2),
        epsilon_p=0.0,
    )

    with pytest.raises(ValueError, match="shock dimension"):
        transition_one_period(base_state(), shocks, graph, params)


def test_transition_rejects_state_attention_incompatible_with_graph():
    params = make_parameters()
    state = base_state()
    sparse_graph = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )

    with pytest.raises(ValueError):
        transition_one_period(state, base_shocks(), sparse_graph, params)


def test_transition_does_not_mutate_inherited_state_or_shocks():
    params = make_parameters()
    graph = complete_graph_three()
    state = base_state()
    shocks = base_shocks()

    beliefs_before = state.beliefs.copy()
    positions_before = state.positions.copy()
    reputation_before = state.reputation.copy()
    attention_before = state.attention.copy()
    epsilon_s_before = shocks.epsilon_s.copy()
    epsilon_b_before = shocks.epsilon_b.copy()

    transition_one_period(state, shocks, graph, params)

    assert np.array_equal(state.beliefs, beliefs_before)
    assert np.array_equal(state.positions, positions_before)
    assert np.array_equal(state.reputation, reputation_before)
    assert np.array_equal(state.attention, attention_before)
    assert np.array_equal(shocks.epsilon_s, epsilon_s_before)
    assert np.array_equal(shocks.epsilon_b, epsilon_b_before)
