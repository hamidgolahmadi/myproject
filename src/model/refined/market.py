"""Refined price and return mappings for Equations (74)-(77)."""

from __future__ import annotations

import numpy as np


def _finite_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def price_change(
    *,
    previous_price: float,
    fundamental_value: float,
    net_order_flow: float,
    chi: float,
    lambda_price: float,
    sigma_p: float,
    epsilon_p: float,
) -> float:
    """Return ``p_t - p_{t-1}`` from Equation (74).

    ``epsilon_p`` is the already-realised standard-normal innovation stored in
    ``PeriodShocks``; only ``sigma_p`` scales it here.
    """

    previous_price = _finite_scalar("previous_price", previous_price)
    fundamental_value = _finite_scalar("fundamental_value", fundamental_value)
    net_order_flow = _finite_scalar("net_order_flow", net_order_flow)
    chi = _finite_scalar("chi", chi)
    lambda_price = _finite_scalar("lambda_price", lambda_price)
    sigma_p = _finite_scalar("sigma_p", sigma_p)
    epsilon_p = _finite_scalar("epsilon_p", epsilon_p)

    if chi < 0.0:
        raise ValueError("chi must be non-negative")
    if lambda_price < 0.0:
        raise ValueError("lambda_price must be non-negative")
    if sigma_p < 0.0:
        raise ValueError("sigma_p must be non-negative")

    return float(
        chi * (fundamental_value - previous_price)
        + lambda_price * net_order_flow
        + sigma_p * epsilon_p
    )


def update_price(
    *,
    previous_price: float,
    fundamental_value: float,
    net_order_flow: float,
    chi: float,
    lambda_price: float,
    sigma_p: float,
    epsilon_p: float,
) -> float:
    """Return the current price ``p_t`` using Equation (75)."""

    previous_price = _finite_scalar("previous_price", previous_price)
    delta_price = price_change(
        previous_price=previous_price,
        fundamental_value=fundamental_value,
        net_order_flow=net_order_flow,
        chi=chi,
        lambda_price=lambda_price,
        sigma_p=sigma_p,
        epsilon_p=epsilon_p,
    )
    return float(previous_price + delta_price)


def market_return(*, previous_price: float, current_price: float) -> float:
    """Return ``r_t = p_t - p_{t-1}`` under the fixed Equation (77) convention."""

    previous_price = _finite_scalar("previous_price", previous_price)
    current_price = _finite_scalar("current_price", current_price)
    return float(current_price - previous_price)
