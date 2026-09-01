"""Unit tests for refined price, return, profit, and reputation equations."""

import numpy as np
import pytest

from src.model.refined import (
    market_return,
    price_change,
    realised_profits,
    update_price,
    update_reputation,
)


def test_price_change_matches_equation_74_exactly():
    delta = price_change(
        previous_price=1.0,
        fundamental_value=1.5,
        net_order_flow=2.0,
        chi=0.2,
        lambda_price=0.05,
        sigma_p=0.1,
        epsilon_p=-1.0,
    )
    assert delta == pytest.approx(0.1)


def test_update_price_matches_equation_75_and_equation_74():
    kwargs = dict(
        previous_price=1.2,
        fundamental_value=1.5,
        net_order_flow=-0.4,
        chi=0.3,
        lambda_price=0.2,
        sigma_p=0.05,
        epsilon_p=0.6,
    )
    current_price = update_price(**kwargs)
    delta = price_change(**kwargs)
    direct = (
        (1.0 - kwargs["chi"]) * kwargs["previous_price"]
        + kwargs["chi"] * kwargs["fundamental_value"]
        + kwargs["lambda_price"] * kwargs["net_order_flow"]
        + kwargs["sigma_p"] * kwargs["epsilon_p"]
    )
    assert current_price == pytest.approx(kwargs["previous_price"] + delta)
    assert current_price == pytest.approx(direct)


def test_fundamental_anchor_pulls_price_toward_value():
    upward = price_change(
        previous_price=1.0,
        fundamental_value=1.5,
        net_order_flow=0.0,
        chi=0.4,
        lambda_price=0.0,
        sigma_p=0.0,
        epsilon_p=0.0,
    )
    downward = price_change(
        previous_price=1.5,
        fundamental_value=1.0,
        net_order_flow=0.0,
        chi=0.4,
        lambda_price=0.0,
        sigma_p=0.0,
        epsilon_p=0.0,
    )
    assert upward > 0.0
    assert downward < 0.0
    assert upward == pytest.approx(-downward)


def test_signed_order_flow_moves_price_in_corresponding_direction():
    buy_pressure = price_change(
        previous_price=1.0,
        fundamental_value=1.0,
        net_order_flow=3.0,
        chi=0.0,
        lambda_price=0.2,
        sigma_p=0.0,
        epsilon_p=0.0,
    )
    sell_pressure = price_change(
        previous_price=1.0,
        fundamental_value=1.0,
        net_order_flow=-3.0,
        chi=0.0,
        lambda_price=0.2,
        sigma_p=0.0,
        epsilon_p=0.0,
    )
    assert buy_pressure == pytest.approx(0.6)
    assert sell_pressure == pytest.approx(-0.6)


def test_price_noise_is_scaled_once_by_sigma_p():
    delta = price_change(
        previous_price=1.0,
        fundamental_value=1.0,
        net_order_flow=0.0,
        chi=0.0,
        lambda_price=0.0,
        sigma_p=0.25,
        epsilon_p=2.0,
    )
    assert delta == pytest.approx(0.5)


def test_market_return_is_price_change_not_simple_return():
    result = market_return(previous_price=2.0, current_price=2.5)
    assert result == pytest.approx(0.5)
    assert result != pytest.approx((2.5 - 2.0) / 2.0)


def test_realised_profit_uses_inherited_positions():
    inherited = np.array([2.0, -1.0, 0.5])
    profits = realised_profits(inherited_positions=inherited, return_=0.3)
    assert np.allclose(profits, np.array([0.6, -0.3, 0.15]))


def test_reputation_update_matches_equation_79():
    previous = np.array([1.0, -1.0])
    profits = np.array([0.5, 0.5])
    updated = update_reputation(
        previous_reputation=previous,
        profits=profits,
        gamma_R=0.8,
    )
    assert np.allclose(updated, np.array([0.9, -0.7]))


def test_zero_reputation_persistence_uses_current_profit_only():
    previous = np.array([10.0, -10.0])
    profits = np.array([0.4, -0.2])
    updated = update_reputation(
        previous_reputation=previous,
        profits=profits,
        gamma_R=0.0,
    )
    assert np.allclose(updated, profits)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("chi", -0.1),
        ("lambda_price", -0.1),
        ("sigma_p", -0.1),
    ],
)
def test_price_change_rejects_negative_scale_parameters(name, value):
    kwargs = dict(
        previous_price=1.0,
        fundamental_value=1.0,
        net_order_flow=0.0,
        chi=0.1,
        lambda_price=0.1,
        sigma_p=0.1,
        epsilon_p=0.0,
    )
    kwargs[name] = value
    with pytest.raises(ValueError):
        price_change(**kwargs)


@pytest.mark.parametrize("gamma_R", [-0.1, 1.0])
def test_reputation_rejects_invalid_persistence(gamma_R):
    with pytest.raises(ValueError):
        update_reputation(
            previous_reputation=np.zeros(2),
            profits=np.ones(2),
            gamma_R=gamma_R,
        )


def test_reputation_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        update_reputation(
            previous_reputation=np.zeros(2),
            profits=np.ones(3),
            gamma_R=0.8,
        )
