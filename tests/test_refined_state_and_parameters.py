"""Unit tests for the first refined-model scaffold and structural objects."""

import numpy as np
import pytest

from src.model.refined import (
    RefinedParameters,
    build_neighbourhoods,
    initialise_state,
    validate_attention,
    validate_graph_support,
)


def make_parameters(**overrides):
    values = dict(
        rho_theta=0.9,
        sigma_theta=0.1,
        v_bar=1.0,
        psi=1.5,
        sigma_s=0.2,
        sigma_b=0.05,
        alpha=0.4,
        kappa=2.0,
        x_bar=3.0,
        chi=0.2,
        lambda_price=0.05,
        sigma_p=0.01,
        gamma_R=0.8,
        beta=1.0,
        sigma_0=1e-3,
    )
    values.update(overrides)
    return RefinedParameters(**values)


def test_refined_parameters_accept_valid_homogeneous_benchmark():
    params = make_parameters()
    assert params.alpha == 0.4
    assert params.x_bar == 3.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("rho_theta", 1.0),
        ("sigma_theta", -0.1),
        ("psi", 0.0),
        ("sigma_s", -0.1),
        ("sigma_b", -0.1),
        ("alpha", 1.0),
        ("alpha", -0.1),
        ("kappa", 0.0),
        ("x_bar", 0.0),
        ("chi", -0.1),
        ("lambda_price", -0.1),
        ("sigma_p", -0.1),
        ("gamma_R", 1.0),
        ("gamma_R", -0.1),
        ("beta", -0.1),
        ("sigma_0", 0.0),
    ],
)
def test_refined_parameters_reject_invalid_values(name, value):
    with pytest.raises(ValueError):
        make_parameters(**{name: value})


def test_graph_attention_separation_and_neighbourhoods():
    graph = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    attention = np.array(
        [
            [0.0, 0.25, 0.75],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    checked_graph = validate_graph_support(graph)
    checked_attention = validate_attention(attention, checked_graph)
    neighbourhoods, degrees = build_neighbourhoods(checked_graph)

    assert np.array_equal(checked_graph, graph)
    assert np.allclose(checked_attention, attention)
    assert np.array_equal(neighbourhoods[0], np.array([1, 2]))
    assert np.array_equal(degrees, np.array([2, 1, 1]))
    assert not np.array_equal(checked_graph.astype(float), checked_attention)


def test_graph_rejects_nonbinary_or_empty_neighbourhood():
    with pytest.raises(ValueError):
        validate_graph_support(np.array([[0.0, 0.5], [1.0, 0.0]]))

    with pytest.raises(ValueError):
        validate_graph_support(np.array([[0, 0], [1, 0]]))


def test_attention_support_and_row_sums():
    graph = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError):
        validate_attention(np.array([[0.1, 0.9], [1.0, 0.0]]), graph)

    with pytest.raises(ValueError):
        validate_attention(np.array([[0.0, 0.8], [1.0, 0.0]]), graph)

    with pytest.raises(ValueError):
        validate_attention(np.array([[0.0, 1.0], [-0.1, 1.1]]), graph)


def test_initial_state_enforces_attention_and_inventory_invariants():
    graph = np.array([[0, 1], [1, 0]])
    attention = np.array([[0.0, 1.0], [1.0, 0.0]])

    state = initialise_state(
        theta=0.0,
        beliefs=np.array([0.1, -0.2]),
        positions=np.array([0.5, -0.5]),
        price=1.0,
        reputation=np.zeros(2),
        attention=attention,
        graph=graph,
        x_bar=1.0,
    )
    assert state.n_agents == 2

    with pytest.raises(ValueError):
        initialise_state(
            theta=0.0,
            beliefs=np.array([0.1, -0.2]),
            positions=np.array([1.1, -0.5]),
            price=1.0,
            reputation=np.zeros(2),
            attention=attention,
            graph=graph,
            x_bar=1.0,
        )
