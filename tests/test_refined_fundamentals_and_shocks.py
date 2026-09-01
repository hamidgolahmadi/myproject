"""Unit tests for PeriodShocks and Equations (42)-(46)."""

import numpy as np
import pytest

from src.model.refined import (
    PeriodShocks,
    RefinedParameters,
    fundamental_value,
    private_signals,
    stationary_fundamental_variance,
    update_fundamental,
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


def test_period_shocks_store_one_realised_bundle():
    signal_noise = np.array([0.10, -0.20, 0.05])
    belief_noise = np.array([0.01, 0.00, -0.01])

    shocks = PeriodShocks(
        u_theta=0.03,
        epsilon_s=signal_noise,
        epsilon_b=belief_noise,
        epsilon_p=-0.5,
    )

    assert shocks.n_agents == 3
    assert shocks.u_theta == 0.03
    assert shocks.epsilon_p == -0.5
    assert np.allclose(shocks.epsilon_s, signal_noise)
    assert np.allclose(shocks.epsilon_b, belief_noise)

    # Construction copies caller-owned arrays, preventing accidental aliasing.
    signal_noise[0] = 99.0
    belief_noise[0] = 99.0
    assert shocks.epsilon_s[0] == pytest.approx(0.10)
    assert shocks.epsilon_b[0] == pytest.approx(0.01)


def test_period_shocks_reject_mismatched_agent_dimensions():
    with pytest.raises(ValueError):
        PeriodShocks(
            u_theta=0.0,
            epsilon_s=np.zeros(3),
            epsilon_b=np.zeros(2),
            epsilon_p=0.0,
        )


def test_period_shocks_reject_empty_or_nonfinite_values():
    with pytest.raises(ValueError):
        PeriodShocks(
            u_theta=0.0,
            epsilon_s=np.array([]),
            epsilon_b=np.array([]),
            epsilon_p=0.0,
        )

    with pytest.raises(ValueError):
        PeriodShocks(
            u_theta=np.nan,
            epsilon_s=np.zeros(2),
            epsilon_b=np.zeros(2),
            epsilon_p=0.0,
        )

    with pytest.raises(ValueError):
        PeriodShocks(
            u_theta=0.0,
            epsilon_s=np.array([0.0, np.inf]),
            epsilon_b=np.zeros(2),
            epsilon_p=0.0,
        )


def test_fundamental_update_matches_equation_42():
    params = make_parameters(rho_theta=0.8)
    theta_t = update_fundamental(theta_previous=0.5, u_theta=-0.1, parameters=params)
    assert theta_t == pytest.approx(0.3)


def test_stationary_fundamental_variance_matches_equation_43():
    params = make_parameters(rho_theta=0.6, sigma_theta=0.2)
    expected = 0.2**2 / (1.0 - 0.6**2)
    assert stationary_fundamental_variance(params) == pytest.approx(expected)


def test_fundamental_value_matches_equation_44():
    params = make_parameters(v_bar=2.0, psi=1.25)
    assert fundamental_value(theta=0.4, parameters=params) == pytest.approx(2.5)


def test_private_signals_match_equations_45_and_46():
    epsilon_s = np.array([0.2, -0.1, 0.0])
    signals = private_signals(theta=0.7, epsilon_s=epsilon_s)
    assert np.allclose(signals, np.array([0.9, 0.6, 0.7]))


def test_private_signals_reject_invalid_noise_shape_or_values():
    with pytest.raises(ValueError):
        private_signals(theta=0.0, epsilon_s=np.zeros((2, 1)))

    with pytest.raises(ValueError):
        private_signals(theta=0.0, epsilon_s=np.array([]))

    with pytest.raises(ValueError):
        private_signals(theta=0.0, epsilon_s=np.array([0.0, np.nan]))


def test_deterministic_information_block_uses_supplied_shocks_exactly():
    params = make_parameters(rho_theta=0.5, v_bar=1.0, psi=2.0)
    shocks = PeriodShocks(
        u_theta=0.2,
        epsilon_s=np.array([0.1, -0.3]),
        epsilon_b=np.zeros(2),
        epsilon_p=0.0,
    )

    theta_t = update_fundamental(0.4, shocks.u_theta, params)
    value_t = fundamental_value(theta_t, params)
    signals_t = private_signals(theta_t, shocks.epsilon_s)

    assert theta_t == pytest.approx(0.4)
    assert value_t == pytest.approx(1.8)
    assert np.allclose(signals_t, np.array([0.5, 0.1]))
