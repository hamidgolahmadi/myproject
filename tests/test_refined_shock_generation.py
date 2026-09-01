from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.model.refined import (
    PeriodShocks,
    RefinedParameters,
    generate_shock_path,
    initialise_state,
    simulate_shock_path,
)


def make_parameters(**overrides: float) -> RefinedParameters:
    values: dict[str, float] = {
        "rho_theta": 0.6,
        "sigma_theta": 0.2,
        "v_bar": 1.0,
        "psi": 0.8,
        "sigma_s": 0.3,
        "sigma_b": 0.1,
        "alpha": 0.4,
        "kappa": 0.7,
        "x_bar": 2.0,
        "chi": 0.2,
        "lambda_price": 0.05,
        "sigma_p": 0.15,
        "gamma_R": 0.8,
        "beta": 1.2,
        "sigma_0": 0.05,
    }
    values.update(overrides)
    return RefinedParameters(**values)


def assert_same_shock_path(
    left: tuple[PeriodShocks, ...],
    right: tuple[PeriodShocks, ...],
) -> None:
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.u_theta == b.u_theta
        np.testing.assert_array_equal(a.epsilon_s, b.epsilon_s)
        np.testing.assert_array_equal(a.epsilon_b, b.epsilon_b)
        assert a.epsilon_p == b.epsilon_p


def uniform_attention(graph: np.ndarray) -> np.ndarray:
    graph = np.asarray(graph, dtype=float)
    return graph / graph.sum(axis=1, keepdims=True)


def test_generate_shock_path_shapes_and_types() -> None:
    path = generate_shock_path(
        n_periods=5,
        n_agents=4,
        parameters=make_parameters(),
        shock_seed=12345,
    )

    assert isinstance(path, tuple)
    assert len(path) == 5
    assert all(isinstance(shock, PeriodShocks) for shock in path)
    assert all(shock.n_agents == 4 for shock in path)
    assert all(shock.epsilon_s.shape == (4,) for shock in path)
    assert all(shock.epsilon_b.shape == (4,) for shock in path)


def test_generate_shock_path_same_seed_is_exactly_reproducible() -> None:
    parameters = make_parameters()
    first = generate_shock_path(
        n_periods=8,
        n_agents=3,
        parameters=parameters,
        shock_seed=9876,
    )
    second = generate_shock_path(
        n_periods=8,
        n_agents=3,
        parameters=parameters,
        shock_seed=9876,
    )

    assert_same_shock_path(first, second)


def test_generate_shock_path_different_seed_changes_path() -> None:
    parameters = make_parameters()
    first = generate_shock_path(
        n_periods=6,
        n_agents=3,
        parameters=parameters,
        shock_seed=101,
    )
    second = generate_shock_path(
        n_periods=6,
        n_agents=3,
        parameters=parameters,
        shock_seed=102,
    )

    identical_periods = []
    for a, b in zip(first, second, strict=True):
        identical_periods.append(
            a.u_theta == b.u_theta
            and np.array_equal(a.epsilon_s, b.epsilon_s)
            and np.array_equal(a.epsilon_b, b.epsilon_b)
            and a.epsilon_p == b.epsilon_p
        )
    assert not all(identical_periods)


def test_component_streams_are_stable_under_signal_scale_change() -> None:
    base = make_parameters(sigma_s=0.25)
    doubled_signal_scale = replace(base, sigma_s=0.5)

    first = generate_shock_path(
        n_periods=7,
        n_agents=5,
        parameters=base,
        shock_seed=444,
    )
    second = generate_shock_path(
        n_periods=7,
        n_agents=5,
        parameters=doubled_signal_scale,
        shock_seed=444,
    )

    for a, b in zip(first, second, strict=True):
        assert a.u_theta == b.u_theta
        np.testing.assert_allclose(b.epsilon_s, 2.0 * a.epsilon_s, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(a.epsilon_b, b.epsilon_b)
        assert a.epsilon_p == b.epsilon_p


def test_price_innovation_is_standardised_and_not_scaled_in_generator() -> None:
    low_price_noise = make_parameters(sigma_p=0.01)
    high_price_noise = replace(low_price_noise, sigma_p=3.0)

    first = generate_shock_path(
        n_periods=6,
        n_agents=4,
        parameters=low_price_noise,
        shock_seed=555,
    )
    second = generate_shock_path(
        n_periods=6,
        n_agents=4,
        parameters=high_price_noise,
        shock_seed=555,
    )

    assert_same_shock_path(first, second)


def test_zero_nonprice_scales_generate_zero_nonprice_shocks() -> None:
    parameters = make_parameters(sigma_theta=0.0, sigma_s=0.0, sigma_b=0.0)
    path = generate_shock_path(
        n_periods=5,
        n_agents=3,
        parameters=parameters,
        shock_seed=777,
    )

    for shock in path:
        assert shock.u_theta == 0.0
        np.testing.assert_array_equal(shock.epsilon_s, np.zeros(3))
        np.testing.assert_array_equal(shock.epsilon_b, np.zeros(3))
        assert np.isfinite(shock.epsilon_p)


def test_generated_path_reuses_common_random_numbers_across_alpha_zero_topologies() -> None:
    parameters = make_parameters(alpha=0.0)
    graph_a = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=int,
    )
    graph_b = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=int,
    )

    common = dict(
        theta=0.15,
        beliefs=np.array([0.2, -0.1, 0.05]),
        positions=np.array([0.3, -0.2, 0.1]),
        price=1.1,
        reputation=np.array([0.05, -0.02, 0.01]),
        x_bar=parameters.x_bar,
    )
    state_a = initialise_state(
        **common,
        attention=uniform_attention(graph_a),
        graph=graph_a,
    )
    state_b = initialise_state(
        **common,
        attention=uniform_attention(graph_b),
        graph=graph_b,
    )

    shock_path = generate_shock_path(
        n_periods=9,
        n_agents=3,
        parameters=parameters,
        shock_seed=24680,
    )

    result_a = simulate_shock_path(state_a, shock_path, graph_a, parameters)
    result_b = simulate_shock_path(state_b, shock_path, graph_b, parameters)

    for state_left, state_right in zip(result_a.states, result_b.states, strict=True):
        assert state_left.theta == state_right.theta
        np.testing.assert_allclose(state_left.beliefs, state_right.beliefs)
        np.testing.assert_allclose(state_left.positions, state_right.positions)
        assert state_left.price == pytest.approx(state_right.price)
        np.testing.assert_allclose(state_left.reputation, state_right.reputation)

    for out_left, out_right in zip(
        result_a.period_outputs,
        result_b.period_outputs,
        strict=True,
    ):
        assert out_left.fundamental_value == out_right.fundamental_value
        np.testing.assert_allclose(out_left.signals, out_right.signals)
        np.testing.assert_allclose(out_left.perceived_values, out_right.perceived_values)
        np.testing.assert_allclose(out_left.valuation_gaps, out_right.valuation_gaps)
        np.testing.assert_allclose(out_left.desired_actions, out_right.desired_actions)
        np.testing.assert_allclose(out_left.actions, out_right.actions)
        assert out_left.net_order_flow == pytest.approx(out_right.net_order_flow)
        assert out_left.return_ == pytest.approx(out_right.return_)
        np.testing.assert_allclose(out_left.profits, out_right.profits)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("n_periods", 0, ValueError),
        ("n_periods", 1.5, TypeError),
        ("n_agents", 0, ValueError),
        ("n_agents", True, TypeError),
    ],
)
def test_generate_shock_path_rejects_invalid_dimensions(
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "n_periods": 3,
        "n_agents": 2,
        "parameters": make_parameters(),
        "shock_seed": 1,
    }
    arguments[field] = value
    with pytest.raises(error_type):
        generate_shock_path(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("shock_seed", "error_type"),
    [
        (-1, ValueError),
        (1.25, TypeError),
        (True, TypeError),
    ],
)
def test_generate_shock_path_rejects_invalid_shock_seed(
    shock_seed: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        generate_shock_path(
            n_periods=3,
            n_agents=2,
            parameters=make_parameters(),
            shock_seed=shock_seed,  # type: ignore[arg-type]
        )


def test_generate_shock_path_requires_refined_parameters() -> None:
    with pytest.raises(TypeError, match="RefinedParameters"):
        generate_shock_path(
            n_periods=3,
            n_agents=2,
            parameters=object(),  # type: ignore[arg-type]
            shock_seed=7,
        )
