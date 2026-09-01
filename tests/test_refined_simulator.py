import numpy as np
import pytest

from src.model.refined import (
    PeriodOutputs,
    PeriodShocks,
    RefinedParameters,
    RefinedState,
    SimulationResult,
    simulate_shock_path,
    transition_one_period,
)


def make_parameters(**overrides) -> RefinedParameters:
    values = dict(
        rho_theta=0.6,
        sigma_theta=0.2,
        v_bar=1.0,
        psi=0.8,
        sigma_s=0.1,
        sigma_b=0.05,
        alpha=0.4,
        kappa=1.2,
        x_bar=2.0,
        chi=0.3,
        lambda_price=0.2,
        sigma_p=0.1,
        gamma_R=0.7,
        beta=1.5,
        sigma_0=0.25,
    )
    values.update(overrides)
    return RefinedParameters(**values)


def make_graph() -> np.ndarray:
    return np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=int,
    )


def make_state() -> RefinedState:
    return RefinedState(
        theta=0.2,
        beliefs=np.array([0.1, -0.2, 0.3]),
        positions=np.array([0.4, -0.3, 0.2]),
        price=1.1,
        reputation=np.array([0.2, -0.1, 0.4]),
        attention=np.array(
            [
                [0.6, 0.4, 0.0],
                [0.0, 0.7, 0.3],
                [0.5, 0.0, 0.5],
            ]
        ),
    )


def make_shock_path() -> tuple[PeriodShocks, ...]:
    return (
        PeriodShocks(
            u_theta=0.10,
            epsilon_s=np.array([0.02, -0.01, 0.03]),
            epsilon_b=np.array([0.01, -0.02, 0.00]),
            epsilon_p=0.20,
        ),
        PeriodShocks(
            u_theta=-0.05,
            epsilon_s=np.array([-0.02, 0.04, -0.01]),
            epsilon_b=np.array([0.00, 0.01, -0.01]),
            epsilon_p=-0.30,
        ),
        PeriodShocks(
            u_theta=0.03,
            epsilon_s=np.array([0.01, 0.00, -0.02]),
            epsilon_b=np.array([-0.01, 0.00, 0.02]),
            epsilon_p=0.10,
        ),
    )


def assert_state_equal(left: RefinedState, right: RefinedState) -> None:
    assert left.theta == pytest.approx(right.theta)
    assert left.price == pytest.approx(right.price)
    np.testing.assert_allclose(left.beliefs, right.beliefs)
    np.testing.assert_allclose(left.positions, right.positions)
    np.testing.assert_allclose(left.reputation, right.reputation)
    np.testing.assert_allclose(left.attention, right.attention)


def assert_output_equal(left: PeriodOutputs, right: PeriodOutputs) -> None:
    assert left.fundamental_value == pytest.approx(right.fundamental_value)
    assert left.net_order_flow == pytest.approx(right.net_order_flow)
    assert left.return_ == pytest.approx(right.return_)
    np.testing.assert_allclose(left.signals, right.signals)
    np.testing.assert_allclose(left.perceived_values, right.perceived_values)
    np.testing.assert_allclose(left.valuation_gaps, right.valuation_gaps)
    np.testing.assert_allclose(left.desired_actions, right.desired_actions)
    np.testing.assert_allclose(left.actions, right.actions)
    np.testing.assert_allclose(left.profits, right.profits)
    np.testing.assert_allclose(left.reputation_scores, right.reputation_scores)


def test_simulation_result_requires_state_output_alignment():
    state = make_state()
    output = transition_one_period(
        state,
        make_shock_path()[0],
        make_graph(),
        make_parameters(),
    )[1]

    result = SimulationResult(states=(state, state), period_outputs=(output,))
    assert result.n_periods == 1
    assert result.initial_state is state
    assert result.final_state is state

    with pytest.raises(ValueError):
        SimulationResult(states=(state,), period_outputs=(output,))


def test_simulator_matches_manual_repeated_transitions_exactly():
    graph = make_graph()
    parameters = make_parameters()
    initial_state = make_state()
    shock_path = make_shock_path()

    result = simulate_shock_path(initial_state, shock_path, graph, parameters)

    manual_states = [initial_state]
    manual_outputs = []
    state = initial_state
    for shock in shock_path:
        state, output = transition_one_period(state, shock, graph, parameters)
        manual_states.append(state)
        manual_outputs.append(output)

    assert result.n_periods == 3
    assert len(result.states) == 4
    assert len(result.period_outputs) == 3

    for actual, expected in zip(result.states, manual_states):
        assert_state_equal(actual, expected)
    for actual, expected in zip(result.period_outputs, manual_outputs):
        assert_output_equal(actual, expected)


def test_simulator_accepts_generator_shock_path():
    graph = make_graph()
    parameters = make_parameters()
    initial_state = make_state()
    shocks = make_shock_path()

    result = simulate_shock_path(
        initial_state,
        (shock for shock in shocks),
        graph,
        parameters,
    )

    assert result.n_periods == len(shocks)


def test_fixed_attention_benchmark_carries_attention_through_all_periods():
    graph = make_graph()
    parameters = make_parameters()
    initial_state = make_state()

    result = simulate_shock_path(
        initial_state,
        make_shock_path(),
        graph,
        parameters,
        adaptive_attention=False,
    )

    for state in result.states:
        np.testing.assert_allclose(state.attention, initial_state.attention)


def test_alpha_zero_is_network_null_over_multiple_periods():
    parameters = make_parameters(alpha=0.0)
    graph_self = np.eye(3, dtype=int)
    graph_dense = np.ones((3, 3), dtype=int)
    initial_attention = np.eye(3)
    initial_state = RefinedState(
        theta=0.2,
        beliefs=np.array([0.1, -0.2, 0.3]),
        positions=np.array([0.4, -0.3, 0.2]),
        price=1.1,
        reputation=np.array([0.2, -0.1, 0.4]),
        attention=initial_attention,
    )
    shocks = make_shock_path()

    result_self = simulate_shock_path(
        initial_state,
        shocks,
        graph_self,
        parameters,
        adaptive_attention=True,
    )
    result_dense = simulate_shock_path(
        initial_state,
        shocks,
        graph_dense,
        parameters,
        adaptive_attention=True,
    )

    for state_self, state_dense in zip(result_self.states, result_dense.states):
        assert state_self.theta == pytest.approx(state_dense.theta)
        assert state_self.price == pytest.approx(state_dense.price)
        np.testing.assert_allclose(state_self.beliefs, state_dense.beliefs)
        np.testing.assert_allclose(state_self.positions, state_dense.positions)
        np.testing.assert_allclose(state_self.reputation, state_dense.reputation)

    for output_self, output_dense in zip(
        result_self.period_outputs,
        result_dense.period_outputs,
    ):
        assert output_self.fundamental_value == pytest.approx(output_dense.fundamental_value)
        assert output_self.net_order_flow == pytest.approx(output_dense.net_order_flow)
        assert output_self.return_ == pytest.approx(output_dense.return_)
        np.testing.assert_allclose(output_self.signals, output_dense.signals)
        np.testing.assert_allclose(output_self.perceived_values, output_dense.perceived_values)
        np.testing.assert_allclose(output_self.valuation_gaps, output_dense.valuation_gaps)
        np.testing.assert_allclose(output_self.desired_actions, output_dense.desired_actions)
        np.testing.assert_allclose(output_self.actions, output_dense.actions)
        np.testing.assert_allclose(output_self.profits, output_dense.profits)

    assert not np.allclose(result_self.final_state.attention, result_dense.final_state.attention)


def test_simulator_does_not_mutate_initial_state_or_shocks():
    graph = make_graph()
    parameters = make_parameters()
    initial_state = make_state()
    shocks = make_shock_path()

    beliefs_before = initial_state.beliefs.copy()
    positions_before = initial_state.positions.copy()
    reputation_before = initial_state.reputation.copy()
    attention_before = initial_state.attention.copy()
    signal_noise_before = tuple(shock.epsilon_s.copy() for shock in shocks)
    belief_noise_before = tuple(shock.epsilon_b.copy() for shock in shocks)

    simulate_shock_path(initial_state, shocks, graph, parameters)

    np.testing.assert_array_equal(initial_state.beliefs, beliefs_before)
    np.testing.assert_array_equal(initial_state.positions, positions_before)
    np.testing.assert_array_equal(initial_state.reputation, reputation_before)
    np.testing.assert_array_equal(initial_state.attention, attention_before)
    for shock, expected_s, expected_b in zip(shocks, signal_noise_before, belief_noise_before):
        np.testing.assert_array_equal(shock.epsilon_s, expected_s)
        np.testing.assert_array_equal(shock.epsilon_b, expected_b)


def test_simulator_rejects_empty_shock_path():
    with pytest.raises(ValueError, match="at least one period"):
        simulate_shock_path(make_state(), (), make_graph(), make_parameters())


def test_simulator_rejects_non_period_shock_elements():
    with pytest.raises(TypeError, match="only PeriodShocks"):
        simulate_shock_path(
            make_state(),
            [make_shock_path()[0], object()],
            make_graph(),
            make_parameters(),
        )


def test_simulator_rejects_shock_dimension_mismatch():
    bad_shock = PeriodShocks(
        u_theta=0.1,
        epsilon_s=np.array([0.0, 0.0]),
        epsilon_b=np.array([0.0, 0.0]),
        epsilon_p=0.0,
    )

    with pytest.raises(ValueError, match="shock dimension"):
        simulate_shock_path(
            make_state(),
            [bad_shock],
            make_graph(),
            make_parameters(),
        )


def test_simulator_rejects_invalid_adaptive_attention_flag():
    with pytest.raises(TypeError, match="adaptive_attention"):
        simulate_shock_path(
            make_state(),
            make_shock_path(),
            make_graph(),
            make_parameters(),
            adaptive_attention=1,
        )
