"""Tests for Equations (48)-(50) of the refined model."""

import numpy as np
import pytest

from src.model.refined import belief_noise_covariance, update_beliefs


def test_belief_noise_covariance_matches_homoskedastic_baseline():
    covariance = belief_noise_covariance(3, sigma_b=0.2)
    assert np.allclose(covariance, 0.04 * np.eye(3))


def test_belief_noise_covariance_accepts_zero_noise():
    covariance = belief_noise_covariance(2, sigma_b=0.0)
    assert np.array_equal(covariance, np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("n_agents", "sigma_b"),
    [
        (0, 0.1),
        (-1, 0.1),
        (2, -0.1),
        (2, np.inf),
    ],
)
def test_belief_noise_covariance_rejects_invalid_inputs(n_agents, sigma_b):
    with pytest.raises(ValueError):
        belief_noise_covariance(n_agents, sigma_b)


def test_update_beliefs_matches_equation_50_exactly():
    signals = np.array([1.0, -0.5, 0.25])
    lagged_beliefs = np.array([0.2, 0.4, -0.1])
    lagged_attention = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.0, 0.0],
        ]
    )
    epsilon_b = np.array([0.01, -0.02, 0.03])
    alpha = 0.4

    expected = (
        (1.0 - alpha) * signals
        + alpha * (lagged_attention @ lagged_beliefs)
        + epsilon_b
    )

    actual = update_beliefs(
        signals,
        lagged_beliefs,
        lagged_attention,
        epsilon_b,
        alpha,
    )

    assert np.allclose(actual, expected)


def test_alpha_zero_removes_network_propagation_exactly():
    signals = np.array([0.3, -0.1])
    lagged_beliefs = np.array([100.0, -100.0])
    epsilon_b = np.array([0.02, -0.03])

    attention_a = np.array([[0.0, 1.0], [1.0, 0.0]])
    attention_b = np.eye(2)

    beliefs_a = update_beliefs(
        signals,
        lagged_beliefs,
        attention_a,
        epsilon_b,
        alpha=0.0,
    )
    beliefs_b = update_beliefs(
        signals,
        lagged_beliefs,
        attention_b,
        epsilon_b,
        alpha=0.0,
    )

    expected = signals + epsilon_b
    assert np.array_equal(beliefs_a, expected)
    assert np.array_equal(beliefs_b, expected)
    assert np.array_equal(beliefs_a, beliefs_b)


def test_belief_update_uses_supplied_lagged_attention_not_a_fixed_point():
    signals = np.array([0.0, 0.0])
    lagged_beliefs = np.array([1.0, 3.0])
    lagged_attention = np.array([[0.0, 1.0], [1.0, 0.0]])
    epsilon_b = np.zeros(2)

    actual = update_beliefs(
        signals,
        lagged_beliefs,
        lagged_attention,
        epsilon_b,
        alpha=0.5,
    )

    expected_lagged_rule = np.array([1.5, 0.5])
    assert np.allclose(actual, expected_lagged_rule)


def test_update_beliefs_rejects_invalid_alpha_and_dimensions():
    signals = np.array([0.1, 0.2])
    lagged_beliefs = np.array([0.0, 0.0])
    attention = np.eye(2)
    epsilon_b = np.zeros(2)

    with pytest.raises(ValueError):
        update_beliefs(signals, lagged_beliefs, attention, epsilon_b, alpha=-0.1)

    with pytest.raises(ValueError):
        update_beliefs(signals, lagged_beliefs, attention, epsilon_b, alpha=1.0)

    with pytest.raises(ValueError):
        update_beliefs(
            signals,
            np.array([0.0, 0.0, 0.0]),
            attention,
            epsilon_b,
            alpha=0.2,
        )

    with pytest.raises(ValueError):
        update_beliefs(
            signals,
            lagged_beliefs,
            np.eye(3),
            epsilon_b,
            alpha=0.2,
        )

    with pytest.raises(ValueError):
        update_beliefs(
            signals,
            lagged_beliefs,
            attention,
            np.zeros(3),
            alpha=0.2,
        )
