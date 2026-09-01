"""Lagged private-social belief updating for Equations (48)-(50)."""

from __future__ import annotations

import numpy as np


def _belief_vector(name: str, value: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _belief_matrix(name: str, value: np.ndarray, n: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def belief_noise_covariance(n_agents: int, sigma_b: float) -> np.ndarray:
    """Return the homogeneous baseline covariance in Equation (49).

    In the first-stage benchmark, belief-processing disturbances are
    cross-sectionally independent and homoskedastic, so
    ``Sigma_epsilon_b = sigma_b**2 * I``.
    """

    if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)):
        raise ValueError("n_agents must be an integer")
    if n_agents < 1:
        raise ValueError("n_agents must be strictly positive")
    if not np.isfinite(sigma_b) or sigma_b < 0.0:
        raise ValueError("sigma_b must be finite and non-negative")

    return (float(sigma_b) ** 2) * np.eye(int(n_agents), dtype=float)


def update_beliefs(
    signals: np.ndarray,
    lagged_beliefs: np.ndarray,
    lagged_attention: np.ndarray,
    epsilon_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply the lagged DeGroot-type update in Equations (48) and (50).

    The runtime rule is

    ``b_t = (1-alpha) s_t + alpha W_{t-1} b_{t-1} + epsilon_b_t``.

    ``lagged_attention`` is deliberately an inherited matrix. This function
    does not solve a contemporaneous fixed point and contains no matrix
    inverse. The realised ``epsilon_b`` vector is supplied explicitly so
    random-number generation remains outside the economic transition logic.
    """

    if not np.isfinite(alpha) or alpha < 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")

    signals_array = _belief_vector("signals", signals)
    n = signals_array.size
    if n == 0:
        raise ValueError("belief update must contain at least one agent")

    lagged_beliefs_array = _belief_vector("lagged_beliefs", lagged_beliefs, n)
    epsilon_b_array = _belief_vector("epsilon_b", epsilon_b, n)
    lagged_attention_array = _belief_matrix("lagged_attention", lagged_attention, n)

    return (
        (1.0 - float(alpha)) * signals_array
        + float(alpha) * (lagged_attention_array @ lagged_beliefs_array)
        + epsilon_b_array
    )
