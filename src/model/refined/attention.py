"""Graph-supported reputation-sensitive attention for Equations (57)-(60)."""

from __future__ import annotations

import numpy as np

from .state import build_neighbourhoods, validate_attention, validate_graph_support


def _reputation_vector(reputation: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(reputation, dtype=float)
    if values.shape != (n,):
        raise ValueError(f"reputation must have shape ({n},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("reputation must contain only finite values")
    return values.copy()


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def local_reputation_statistics(
    reputation: np.ndarray,
    graph: np.ndarray,
    sigma_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local reputation means and regularised dispersions, Eqs. (57)-(58).

    Statistics for agent ``i`` are computed only over its feasible information
    neighbourhood ``N_i``.  The positive ``sigma_0`` floor is inside the square
    root exactly as in Equation (58).
    """

    graph_array = validate_graph_support(graph)
    n = graph_array.shape[0]
    reputation_array = _reputation_vector(reputation, n)
    sigma_0 = _positive_finite("sigma_0", sigma_0)
    neighbourhoods, _ = build_neighbourhoods(graph_array)

    means = np.empty(n, dtype=float)
    dispersions = np.empty(n, dtype=float)

    for i, neighbours in enumerate(neighbourhoods):
        local_reputation = reputation_array[neighbours]
        local_mean = float(np.mean(local_reputation))
        local_variance = float(np.mean((local_reputation - local_mean) ** 2))
        means[i] = local_mean
        dispersions[i] = np.sqrt(local_variance + sigma_0**2)

    return means, dispersions


def standardised_reputation_scores(
    reputation: np.ndarray,
    graph: np.ndarray,
    sigma_0: float,
) -> np.ndarray:
    """Return the graph-supported score matrix ``z_ij,t`` from Equation (59).

    Entries outside the feasible graph support are stored as zero.  They are
    placeholders only and are never passed into the softmax allocation.
    """

    graph_array = validate_graph_support(graph)
    n = graph_array.shape[0]
    reputation_array = _reputation_vector(reputation, n)
    means, dispersions = local_reputation_statistics(
        reputation_array,
        graph_array,
        sigma_0,
    )
    neighbourhoods, _ = build_neighbourhoods(graph_array)

    scores = np.zeros((n, n), dtype=float)
    for i, neighbours in enumerate(neighbourhoods):
        scores[i, neighbours] = (
            reputation_array[neighbours] - means[i]
        ) / dispersions[i]

    return scores


def update_attention(
    reputation: np.ndarray,
    graph: np.ndarray,
    beta: float,
    sigma_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frictionless reputation-sensitive attention, Equation (60).

    The softmax is evaluated only over feasible neighbours.  Unsupported
    weights remain exactly zero.  ``beta = 0`` therefore returns uniform
    graph-supported attention.  A max-shifted softmax is used for numerical
    stability without changing the allocation.

    Returns
    -------
    attention, scores
        The row-stochastic ``W_t`` and the standardised score matrix ``z_t``.
    """

    graph_array = validate_graph_support(graph)
    n = graph_array.shape[0]
    reputation_array = _reputation_vector(reputation, n)
    beta = _nonnegative_finite("beta", beta)
    sigma_0 = _positive_finite("sigma_0", sigma_0)
    neighbourhoods, degrees = build_neighbourhoods(graph_array)
    scores = standardised_reputation_scores(
        reputation_array,
        graph_array,
        sigma_0,
    )

    attention = np.zeros((n, n), dtype=float)

    for i, neighbours in enumerate(neighbourhoods):
        if beta == 0.0:
            attention[i, neighbours] = 1.0 / degrees[i]
            continue

        logits = beta * scores[i, neighbours]
        if not np.all(np.isfinite(logits)):
            raise ValueError("attention logits must be finite")
        shifted = logits - np.max(logits)
        weights = np.exp(shifted)
        attention[i, neighbours] = weights / np.sum(weights)

    return validate_attention(attention, graph_array), scores
