"""State containers and network validation for the refined model.

This module implements the structural objects in Equations (35)-(40) and
keeps persistent state separate from within-period diagnostic outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_FLOAT_TOL = 1e-12


def _as_vector(name: str, value: Iterable[float] | np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _as_matrix(name: str, value: Iterable[Iterable[float]] | np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if n is not None and array.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def validate_graph_support(graph: np.ndarray) -> np.ndarray:
    """Validate the feasible-information graph ``G`` from Equations (35)-(36).

    ``G`` is a directed binary support matrix. Every row must contain at least
    one feasible source. Self-links are not prohibited by Equations (35)-(36),
    so this function does not silently impose a zero diagonal.
    """

    graph_array = np.asarray(graph)
    if graph_array.ndim != 2 or graph_array.shape[0] != graph_array.shape[1]:
        raise ValueError("graph must be a square matrix")
    if graph_array.shape[0] == 0:
        raise ValueError("graph must contain at least one agent")
    if not np.all(np.isin(graph_array, (0, 1))):
        raise ValueError("graph must be binary")

    graph_array = graph_array.astype(np.int8, copy=True)
    degree = graph_array.sum(axis=1)
    if np.any(degree < 1):
        raise ValueError("every agent must have at least one feasible information source")
    return graph_array


def build_neighbourhoods(graph: np.ndarray) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Return feasible source indices and row degrees for Equation (36)."""

    graph_array = validate_graph_support(graph)
    neighbourhoods = tuple(np.flatnonzero(graph_array[i]).astype(int) for i in range(graph_array.shape[0]))
    degrees = graph_array.sum(axis=1).astype(int)
    return neighbourhoods, degrees


def validate_attention(attention: np.ndarray, graph: np.ndarray, *, atol: float = _FLOAT_TOL) -> np.ndarray:
    """Validate effective attention ``W`` against graph support, Equation (38)."""

    graph_array = validate_graph_support(graph)
    attention_array = _as_matrix("attention", attention, graph_array.shape[0])

    if np.any(attention_array < -atol):
        raise ValueError("attention weights must be non-negative")
    attention_array[np.abs(attention_array) <= atol] = 0.0

    if np.any(np.abs(attention_array[graph_array == 0]) > atol):
        raise ValueError("attention assigns weight outside the feasible graph")

    row_sums = attention_array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=atol):
        raise ValueError("each attention row must sum to one")

    return attention_array


@dataclass(frozen=True, slots=True)
class RefinedState:
    """Persistent state ``(theta, b, x, p, R, W)`` at one date."""

    theta: float
    beliefs: np.ndarray
    positions: np.ndarray
    price: float
    reputation: np.ndarray
    attention: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite")
        if not np.isfinite(self.price):
            raise ValueError("price must be finite")

        beliefs = _as_vector("beliefs", self.beliefs)
        n = beliefs.size
        if n == 0:
            raise ValueError("state must contain at least one agent")
        positions = _as_vector("positions", self.positions, n)
        reputation = _as_vector("reputation", self.reputation, n)
        attention = _as_matrix("attention", self.attention, n)

        object.__setattr__(self, "theta", float(self.theta))
        object.__setattr__(self, "price", float(self.price))
        object.__setattr__(self, "beliefs", beliefs)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "reputation", reputation)
        object.__setattr__(self, "attention", attention)

    @property
    def n_agents(self) -> int:
        return int(self.beliefs.size)

    def validate_against(self, graph: np.ndarray, x_bar: float) -> None:
        """Validate support and inventory invariants for an existing state."""

        if x_bar <= 0.0 or not np.isfinite(x_bar):
            raise ValueError("x_bar must be finite and strictly positive")
        graph_array = validate_graph_support(graph)
        if graph_array.shape != (self.n_agents, self.n_agents):
            raise ValueError("graph dimension does not match state dimension")
        validate_attention(self.attention, graph_array)
        if np.any(np.abs(self.positions) > x_bar + _FLOAT_TOL):
            raise ValueError("state positions violate the inventory bound")


@dataclass(frozen=True, slots=True)
class PeriodOutputs:
    """Non-persistent objects produced during one period transition."""

    fundamental_value: float
    signals: np.ndarray
    perceived_values: np.ndarray
    valuation_gaps: np.ndarray
    desired_actions: np.ndarray
    actions: np.ndarray
    net_order_flow: float
    return_: float
    profits: np.ndarray
    reputation_scores: np.ndarray

    def __post_init__(self) -> None:
        scalar_names = ("fundamental_value", "net_order_flow", "return_")
        for name in scalar_names:
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))

        signals = _as_vector("signals", self.signals)
        n = signals.size
        if n == 0:
            raise ValueError("period outputs must contain at least one agent")
        for name in (
            "perceived_values",
            "valuation_gaps",
            "desired_actions",
            "actions",
            "profits",
        ):
            object.__setattr__(self, name, _as_vector(name, getattr(self, name), n))

        scores = np.asarray(self.reputation_scores, dtype=float)
        if scores.shape != (n, n):
            raise ValueError(f"reputation_scores must have shape ({n}, {n})")
        if not np.all(np.isfinite(scores)):
            raise ValueError("reputation_scores must contain only finite values")
        object.__setattr__(self, "reputation_scores", scores.copy())


def initialise_state(
    *,
    theta: float,
    beliefs: np.ndarray,
    positions: np.ndarray,
    price: float,
    reputation: np.ndarray,
    attention: np.ndarray,
    graph: np.ndarray,
    x_bar: float,
) -> RefinedState:
    """Construct and validate the initial state in Equation (40)."""

    state = RefinedState(
        theta=theta,
        beliefs=beliefs,
        positions=positions,
        price=price,
        reputation=reputation,
        attention=attention,
    )
    state.validate_against(graph, x_bar)
    return state
