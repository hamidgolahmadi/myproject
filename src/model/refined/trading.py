"""Perceived values, desired trades, inventory projection, and order flow.

This module implements Equations (63)-(72) of the refined doctoral model.
Desired one-period actions and accumulated inventory constraints are kept as
separate mechanisms, exactly as in the report.
"""

from __future__ import annotations

import numpy as np


def _vector(name: str, value: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one agent")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def perceived_values(beliefs: np.ndarray, *, v_bar: float, psi: float) -> np.ndarray:
    """Map beliefs into perceived fundamental values, Equation (63)."""

    beliefs_array = _vector("beliefs", beliefs)
    v_bar_value = _finite_scalar("v_bar", v_bar)
    psi_value = _finite_scalar("psi", psi)
    if psi_value <= 0.0:
        raise ValueError("psi must be strictly positive")
    return v_bar_value + psi_value * beliefs_array


def valuation_gaps(perceived: np.ndarray, *, lagged_price: float) -> np.ndarray:
    """Compare perceived value with inherited price, Equation (64)."""

    perceived_array = _vector("perceived", perceived)
    price_value = _finite_scalar("lagged_price", lagged_price)
    return perceived_array - price_value


def desired_actions(gaps: np.ndarray, *, kappa: float) -> np.ndarray:
    """Return bounded desired one-period adjustments, Equation (66)."""

    gaps_array = _vector("gaps", gaps)
    kappa_value = _finite_scalar("kappa", kappa)
    if kappa_value <= 0.0:
        raise ValueError("kappa must be strictly positive")
    return np.tanh(kappa_value * gaps_array)


def inventory_feasible_bounds(
    lagged_positions: np.ndarray,
    *,
    x_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper feasible action bounds, Equation (68)."""

    positions = _vector("lagged_positions", lagged_positions)
    x_bar_value = _finite_scalar("x_bar", x_bar)
    if x_bar_value <= 0.0:
        raise ValueError("x_bar must be strictly positive")
    if np.any(np.abs(positions) > x_bar_value + 1e-12):
        raise ValueError("lagged_positions violate the inventory bound")

    lower = -x_bar_value - positions
    upper = x_bar_value - positions
    return lower, upper


def execute_actions(
    desired: np.ndarray,
    lagged_positions: np.ndarray,
    *,
    x_bar: float,
) -> np.ndarray:
    """Project desired actions onto the feasible interval, Equations (69)-(70)."""

    desired_array = _vector("desired", desired)
    positions = _vector("lagged_positions", lagged_positions, desired_array.size)
    lower, upper = inventory_feasible_bounds(positions, x_bar=x_bar)
    return np.minimum(upper, np.maximum(lower, desired_array))


def update_positions(lagged_positions: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Update the stock of positions from signed executed flows, Equation (71)."""

    positions = _vector("lagged_positions", lagged_positions)
    actions_array = _vector("actions", actions, positions.size)
    return positions + actions_array


def net_order_flow(actions: np.ndarray) -> float:
    """Aggregate signed executed actions into net order flow, Equation (72)."""

    actions_array = _vector("actions", actions)
    return float(np.sum(actions_array))
