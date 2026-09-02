"""Realised-influence and common-exposure diagnostics, Section 5.5 Eqs. (251)-(265).

These functions evaluate graph-supported attention matrices already produced by
the refined simulator.  They do not alter the economic transition.  Structural
hubs are always defined from the feasible graph ``G``; they are never selected
from realised attention ``W_t``.

The report defines ``H_q(G)`` as exactly the ``q`` nodes with largest
in-degrees but does not specify how to resolve an in-degree tie at the cutoff.
For reproducibility this implementation ranks by decreasing in-degree and then
by increasing node label.  This is a computational tie-break only, not an
additional economic ranking criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.refined import SimulationResult
from src.model.refined.state import validate_attention, validate_graph_support
from src.topologies.refined.diagnostics import in_degrees


_FLOAT_ATOL = 1e-12


def _positive_q(q: int, n_agents: int) -> int:
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)):
        raise TypeError("q must be an integer")
    q = int(q)
    if not 1 <= q <= n_agents:
        raise ValueError("q must satisfy 1 <= q <= n_agents")
    return q


def _validated_benchmark_graph(graph: np.ndarray) -> np.ndarray:
    graph_array = validate_graph_support(graph)
    if np.any(np.diag(graph_array) != 0):
        raise ValueError("refined benchmark influence metrics require zero self-links")
    return graph_array


def _validated_attention(attention: np.ndarray, graph: np.ndarray) -> np.ndarray:
    return validate_attention(attention, _validated_benchmark_graph(graph))


def structural_hub_nodes(graph: np.ndarray, *, q: int) -> tuple[int, ...]:
    """Return the report's structural hub set ``H_q(G)`` used in Eq. (257).

    Nodes are ranked by decreasing directed in-degree.  Equal in-degrees are
    resolved by increasing node label solely to make the set deterministic.
    """

    graph_array = _validated_benchmark_graph(graph)
    degrees = in_degrees(graph_array)
    q = _positive_q(q, graph_array.shape[0])
    labels = np.arange(graph_array.shape[0], dtype=int)
    order = np.lexsort((labels, -degrees))
    return tuple(int(node) for node in order[:q])


def attention_entropy(attention: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Return row-wise Shannon attention entropy ``H(w_i,t)``.

    Zero attention weights contribute zero by continuity and are never passed
    to ``log``.
    """

    attention_array = _validated_attention(attention, graph)
    positive = attention_array > 0.0
    terms = np.zeros_like(attention_array, dtype=float)
    terms[positive] = attention_array[positive] * np.log(attention_array[positive])
    entropy = -terms.sum(axis=1)
    entropy[np.abs(entropy) <= _FLOAT_ATOL] = 0.0
    return entropy


def normalised_attention_entropy(attention: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Return ``H(w_i,t) / log(d_i^out)`` from Equation (251).

    Equation (251) is explicitly defined for ``d_i^out > 1``.  Rather than
    invent a convention for degree-one rows, this function rejects such a
    graph.
    """

    graph_array = _validated_benchmark_graph(graph)
    degrees = graph_array.sum(axis=1).astype(float)
    if np.any(degrees <= 1.0):
        raise ValueError("Equation (251) requires every out-degree to exceed one")
    values = attention_entropy(attention, graph_array) / np.log(degrees)
    values[np.abs(values) <= _FLOAT_ATOL] = 0.0
    if np.any(values < -_FLOAT_ATOL) or np.any(values > 1.0 + _FLOAT_ATOL):
        raise RuntimeError("normalised attention entropy fell outside [0, 1]")
    return np.clip(values, 0.0, 1.0)


def effective_number_of_sources(attention: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Return ``exp(H(w_i,t))`` from Equation (252)."""

    return np.exp(attention_entropy(attention, graph))


def realised_influence_shares(attention: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Return source shares ``s^I_j,t = sum_i w_ij,t / N``, Eqs. (254)-(255)."""

    attention_array = _validated_attention(attention, graph)
    n_agents = attention_array.shape[0]
    shares = attention_array.sum(axis=0) / n_agents
    if not np.isclose(float(shares.sum()), 1.0, rtol=0.0, atol=_FLOAT_ATOL):
        raise RuntimeError("realised influence shares do not sum to one")
    shares[np.abs(shares) <= _FLOAT_ATOL] = 0.0
    return shares


def realised_influence_hhi(attention: np.ndarray, graph: np.ndarray) -> float:
    """Return the realised-influence Herfindahl index from Equation (256)."""

    shares = realised_influence_shares(attention, graph)
    value = float(np.sum(np.square(shares)))
    n_agents = shares.size
    if value < (1.0 / n_agents) - _FLOAT_ATOL or value > 1.0 + _FLOAT_ATOL:
        raise RuntimeError("realised influence HHI fell outside its probability-share bounds")
    return float(np.clip(value, 1.0 / n_agents, 1.0))


def realised_hub_influence_share(
    attention: np.ndarray,
    graph: np.ndarray,
    *,
    q: int,
) -> float:
    """Return ``S^I_q,t`` for the structural hubs ``H_q(G)``, Equation (257)."""

    shares = realised_influence_shares(attention, graph)
    hubs = structural_hub_nodes(graph, q=q)
    return float(np.sum(shares[np.asarray(hubs, dtype=int)]))


def attention_overlap(attention: np.ndarray, graph: np.ndarray) -> float:
    """Return aggregate pairwise attention overlap, Equations (258)-(264).

    The direct pairwise formula in Equation (259) and the compact matrix formula
    in Equation (260) are both evaluated and required to agree numerically.
    """

    attention_array = _validated_attention(attention, graph)
    n_agents = attention_array.shape[0]
    if n_agents < 2:
        raise ValueError("attention overlap requires at least two agents")

    gram = attention_array @ attention_array.T
    direct = float((gram.sum() - np.trace(gram)) / (n_agents * (n_agents - 1)))

    column_totals = attention_array.T @ np.ones(n_agents, dtype=float)
    compact = float(
        (np.dot(column_totals, column_totals) - np.sum(np.square(attention_array)))
        / (n_agents * (n_agents - 1))
    )
    if not np.isclose(direct, compact, rtol=1e-11, atol=_FLOAT_ATOL):
        raise RuntimeError("Equations (259) and (260) disagree numerically")
    if direct < -_FLOAT_ATOL or direct > 1.0 + _FLOAT_ATOL:
        raise RuntimeError("attention overlap fell outside [0, 1]")
    return float(np.clip(direct, 0.0, 1.0))


def attention_mobility(
    attention: np.ndarray,
    previous_attention: np.ndarray,
    graph: np.ndarray,
) -> float:
    """Return RMS one-period attention reallocation from Equation (265)."""

    current = _validated_attention(attention, graph)
    previous = _validated_attention(previous_attention, graph)
    n_agents = current.shape[0]
    return float(np.linalg.norm(current - previous, ord="fro") / np.sqrt(n_agents))


@dataclass(frozen=True, slots=True)
class RealisedInfluencePoint:
    """Report-defined realised-influence diagnostics for one simulated period."""

    period: int
    normalised_entropies: np.ndarray
    effective_sources: np.ndarray
    mean_normalised_entropy: float
    mean_effective_sources: float
    source_influence_shares: np.ndarray
    influence_hhi: float
    structural_hub_influence_share: float
    attention_overlap: float
    attention_mobility: float

    def __post_init__(self) -> None:
        if isinstance(self.period, bool) or not isinstance(self.period, (int, np.integer)):
            raise TypeError("period must be an integer")
        period = int(self.period)
        if period < 1:
            raise ValueError("period must be strictly positive")
        object.__setattr__(self, "period", period)

        entropies = np.asarray(self.normalised_entropies, dtype=float)
        effective = np.asarray(self.effective_sources, dtype=float)
        shares = np.asarray(self.source_influence_shares, dtype=float)
        if entropies.ndim != 1 or effective.ndim != 1 or shares.ndim != 1:
            raise ValueError("agent/source diagnostic arrays must be one-dimensional")
        if entropies.size == 0 or effective.shape != entropies.shape or shares.shape != entropies.shape:
            raise ValueError("agent/source diagnostic arrays must share a non-empty dimension")
        if not np.all(np.isfinite(entropies)) or not np.all(np.isfinite(effective)) or not np.all(np.isfinite(shares)):
            raise ValueError("agent/source diagnostic arrays must be finite")
        if np.any(entropies < -_FLOAT_ATOL) or np.any(entropies > 1.0 + _FLOAT_ATOL):
            raise ValueError("normalised_entropies must lie in [0, 1]")
        if np.any(effective < 1.0 - _FLOAT_ATOL):
            raise ValueError("effective_sources must be at least one")
        if np.any(shares < -_FLOAT_ATOL) or not np.isclose(shares.sum(), 1.0, rtol=0.0, atol=_FLOAT_ATOL):
            raise ValueError("source_influence_shares must be non-negative and sum to one")

        object.__setattr__(self, "normalised_entropies", entropies.copy())
        object.__setattr__(self, "effective_sources", effective.copy())
        object.__setattr__(self, "source_influence_shares", shares.copy())

        for name in (
            "mean_normalised_entropy",
            "mean_effective_sources",
            "influence_hhi",
            "structural_hub_influence_share",
            "attention_overlap",
            "attention_mobility",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < -_FLOAT_ATOL:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, max(0.0, value))


@dataclass(frozen=True, slots=True)
class RealisedInfluencePath:
    """Structural hub set plus period-by-period realised-influence diagnostics."""

    hub_q: int
    structural_hubs: tuple[int, ...]
    points: tuple[RealisedInfluencePoint, ...]

    def __post_init__(self) -> None:
        if isinstance(self.hub_q, bool) or not isinstance(self.hub_q, (int, np.integer)):
            raise TypeError("hub_q must be an integer")
        hub_q = int(self.hub_q)
        if hub_q < 1:
            raise ValueError("hub_q must be strictly positive")
        if len(self.structural_hubs) != hub_q or len(set(self.structural_hubs)) != hub_q:
            raise ValueError("structural_hubs must contain exactly hub_q unique nodes")
        if len(self.points) == 0 or not all(isinstance(point, RealisedInfluencePoint) for point in self.points):
            raise ValueError("points must contain at least one RealisedInfluencePoint")
        periods = tuple(point.period for point in self.points)
        if periods != tuple(range(1, len(self.points) + 1)):
            raise ValueError("influence points must be contiguous from period one")
        object.__setattr__(self, "hub_q", hub_q)
        object.__setattr__(self, "structural_hubs", tuple(int(node) for node in self.structural_hubs))


def realised_influence_path(
    result: SimulationResult,
    graph: np.ndarray,
    *,
    q: int,
) -> RealisedInfluencePath:
    """Compute Equations (251)-(265) over ``t=1,...,T`` for one simulation.

    Entropy/concentration/overlap at period ``t`` use ``W_t = states[t].attention``.
    Mobility uses the inherited pair ``(W_{t-1}, W_t)`` and therefore naturally
    includes the first transition from neutral ``W_0`` to ``W_1``.
    """

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    graph_array = _validated_benchmark_graph(graph)
    n_agents = graph_array.shape[0]
    if result.initial_state.n_agents != n_agents or any(state.n_agents != n_agents for state in result.states):
        raise ValueError("graph and simulation state dimensions must agree")

    hubs = structural_hub_nodes(graph_array, q=q)
    q = len(hubs)
    points: list[RealisedInfluencePoint] = []

    for period in range(1, result.n_periods + 1):
        current = result.states[period].attention
        previous = result.states[period - 1].attention
        entropies = normalised_attention_entropy(current, graph_array)
        effective = effective_number_of_sources(current, graph_array)
        shares = realised_influence_shares(current, graph_array)

        points.append(
            RealisedInfluencePoint(
                period=period,
                normalised_entropies=entropies,
                effective_sources=effective,
                mean_normalised_entropy=float(np.mean(entropies)),
                mean_effective_sources=float(np.mean(effective)),
                source_influence_shares=shares,
                influence_hhi=realised_influence_hhi(current, graph_array),
                structural_hub_influence_share=float(
                    np.sum(shares[np.asarray(hubs, dtype=int)])
                ),
                attention_overlap=attention_overlap(current, graph_array),
                attention_mobility=attention_mobility(current, previous, graph_array),
            )
        )

    return RealisedInfluencePath(
        hub_q=q,
        structural_hubs=hubs,
        points=tuple(points),
    )
