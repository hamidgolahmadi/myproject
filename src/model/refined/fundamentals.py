"""Fundamental-state, value, and private-signal mappings for Equations (42)-(46)."""

from __future__ import annotations

import numpy as np

from .parameters import RefinedParameters


def update_fundamental(
    theta_previous: float,
    u_theta: float,
    parameters: RefinedParameters,
) -> float:
    """Advance the latent AR(1) fundamental state, Equation (42)."""

    if not np.isfinite(theta_previous):
        raise ValueError("theta_previous must be finite")
    if not np.isfinite(u_theta):
        raise ValueError("u_theta must be finite")
    return float(parameters.rho_theta * theta_previous + u_theta)


def stationary_fundamental_variance(parameters: RefinedParameters) -> float:
    """Return the stationary variance in Equation (43)."""

    return float(parameters.sigma_theta**2 / (1.0 - parameters.rho_theta**2))


def fundamental_value(theta: float, parameters: RefinedParameters) -> float:
    """Map the latent state into asset value, Equation (44)."""

    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    return float(parameters.v_bar + parameters.psi * theta)


def private_signals(theta: float, epsilon_s: np.ndarray) -> np.ndarray:
    """Construct the vector of private signals, Equations (45)-(46).

    ``epsilon_s`` is supplied as an already-realised signal-noise vector.
    This function performs no sampling and introduces no hidden scaling.
    """

    if not np.isfinite(theta):
        raise ValueError("theta must be finite")

    noise = np.asarray(epsilon_s, dtype=float)
    if noise.ndim != 1:
        raise ValueError("epsilon_s must be one-dimensional")
    if noise.size == 0:
        raise ValueError("epsilon_s must contain at least one agent")
    if not np.all(np.isfinite(noise)):
        raise ValueError("epsilon_s must contain only finite values")

    return np.full(noise.shape, float(theta), dtype=float) + noise
