"""Realised-profit and reputation updates for Equations (78)-(79)."""

from __future__ import annotations

import numpy as np


def _finite_vector(name: str, values: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def realised_profits(*, inherited_positions: np.ndarray, return_: float) -> np.ndarray:
    """Return realised profit ``pi_t = x_{t-1} r_t`` from Equation (78).

    The function deliberately accepts inherited positions rather than current
    positions so the current trade cannot earn the contemporaneous price move
    that it helps create.
    """

    positions = _finite_vector("inherited_positions", inherited_positions)
    if positions.size == 0:
        raise ValueError("inherited_positions must contain at least one agent")
    return_ = float(return_)
    if not np.isfinite(return_):
        raise ValueError("return_ must be finite")
    return positions * return_


def update_reputation(
    *,
    previous_reputation: np.ndarray,
    profits: np.ndarray,
    gamma_R: float,
) -> np.ndarray:
    """Return ``R_t`` from the exponentially weighted Equation (79)."""

    previous = _finite_vector("previous_reputation", previous_reputation)
    if previous.size == 0:
        raise ValueError("previous_reputation must contain at least one agent")
    profits_array = _finite_vector("profits", profits, previous.size)

    gamma_R = float(gamma_R)
    if not np.isfinite(gamma_R):
        raise ValueError("gamma_R must be finite")
    if not 0.0 <= gamma_R < 1.0:
        raise ValueError("gamma_R must satisfy 0 <= gamma_R < 1")

    return gamma_R * previous + (1.0 - gamma_R) * profits_array
