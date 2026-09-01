"""Structural diagnostics for refined benchmark graph ensembles, Eqs. (203)-(211).

These diagnostics validate the *feasible* directed graph ``G`` before market
outcomes are interpreted.  Clustering, path length, and component statistics
use the symmetrised support ``G^u`` exactly as specified in Section 5.3.1 of
the doctoral report.  Economic belief updating remains directed; symmetrising
here is diagnostic only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.model.refined.state import validate_graph_support


def _benchmark_graph(graph: np.ndarray) -> np.ndarray:
    """Validate a graph for Section 5.3.1 benchmark diagnostics."""

    graph_array = validate_graph_support(graph)
    if np.any(np.diag(graph_array) != 0):
        raise ValueError("benchmark structural diagnostics require zero self-links")
    return graph_array


def _positive_q(q: int, n_agents: int) -> int:
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)):
        raise TypeError("q must be an integer")
    q = int(q)
    if not 1 <= q <= n_agents:
        raise ValueError("q must satisfy 1 <= q <= n_agents")
    return q


def in_degrees(graph: np.ndarray) -> np.ndarray:
    """Return directed in-degrees ``d_j^in`` as column sums of ``G``."""

    graph_array = _benchmark_graph(graph)
    return graph_array.sum(axis=0, dtype=np.int64)


def in_degree_gini(graph: np.ndarray) -> float:
    """Return the in-degree Gini coefficient from Equations (203)-(205)."""

    degrees = in_degrees(graph).astype(float)
    n_agents = degrees.size
    total_links = float(np.sum(degrees))
    pairwise_absolute_difference = np.abs(
        degrees[:, np.newaxis] - degrees[np.newaxis, :]
    ).sum()
    return float(pairwise_absolute_difference / (2.0 * n_agents * total_links))


def hub_link_share(graph: np.ndarray, q: int) -> float:
    """Return the share of incoming links received by the top-``q`` hubs, Eq. (206)."""

    degrees = in_degrees(graph)
    q = _positive_q(q, degrees.size)
    total_links = int(np.sum(degrees))
    top_q_total = int(np.sort(degrees)[-q:].sum())
    return float(top_q_total / total_links)


def symmetrised_support(graph: np.ndarray) -> np.ndarray:
    """Return the undirected support ``G^u`` from Equation (208)."""

    graph_array = _benchmark_graph(graph)
    support = ((graph_array + graph_array.T) > 0).astype(np.int8)
    np.fill_diagonal(support, 0)
    return support


def global_clustering(graph: np.ndarray) -> float:
    """Return transitivity ``3 * triangles / connected triples``, Eq. (209).

    The calculation is performed on ``G^u``.  If the symmetrised graph contains
    no connected triples, the diagnostic convention is ``0.0``.
    """

    support = symmetrised_support(graph).astype(np.int64)
    undirected_degree = support.sum(axis=1, dtype=np.int64)
    connected_triples = int(np.sum(undirected_degree * (undirected_degree - 1) // 2))
    if connected_triples == 0:
        return 0.0

    triangles = int(np.trace(support @ support @ support) // 6)
    return float((3.0 * triangles) / connected_triples)


def _connected_components(support: np.ndarray) -> tuple[tuple[int, ...], ...]:
    n_agents = support.shape[0]
    unseen = set(range(n_agents))
    components: list[tuple[int, ...]] = []

    while unseen:
        start = min(unseen)
        queue: deque[int] = deque([start])
        unseen.remove(start)
        component: list[int] = []

        while queue:
            node = queue.popleft()
            component.append(node)
            neighbours = np.flatnonzero(support[node])
            for neighbour in neighbours:
                neighbour = int(neighbour)
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)

        components.append(tuple(sorted(component)))

    return tuple(components)


def _largest_component(support: np.ndarray) -> tuple[int, ...]:
    components = _connected_components(support)
    # Deterministic tie-break: components are discovered by smallest node label.
    return max(components, key=lambda component: (len(component), -component[0]))


def largest_component_share(graph: np.ndarray) -> float:
    """Return ``omega_max = n_max / N`` from Equation (211)."""

    support = symmetrised_support(graph)
    largest = _largest_component(support)
    return float(len(largest) / support.shape[0])


def _distances_from_source(
    support: np.ndarray,
    source: int,
    allowed: set[int],
) -> dict[int, int]:
    distances = {source: 0}
    queue: deque[int] = deque([source])

    while queue:
        node = queue.popleft()
        for neighbour in np.flatnonzero(support[node]):
            neighbour = int(neighbour)
            if neighbour in allowed and neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)

    return distances


def average_path_length_lcc(graph: np.ndarray) -> float:
    """Return average path length within the largest component, Equation (210)."""

    support = symmetrised_support(graph)
    largest = _largest_component(support)
    n_max = len(largest)
    if n_max <= 1:
        return 0.0

    allowed = set(largest)
    distance_sum = 0
    for source in largest:
        distances = _distances_from_source(support, source, allowed)
        if len(distances) != n_max:
            raise RuntimeError("largest-component path calculation encountered disconnection")
        distance_sum += sum(
            distance for node, distance in distances.items() if node != source
        )

    return float(distance_sum / (n_max * (n_max - 1)))


@dataclass(frozen=True, slots=True)
class StructuralDiagnostics:
    """One graph's report-defined structural validation statistics."""

    n_agents: int
    total_links: int
    mean_out_degree: float
    in_degree_gini: float
    hub_q: int
    hub_link_share: float
    global_clustering: float
    average_path_length_lcc: float
    largest_component_share: float


def diagnose_graph(graph: np.ndarray, *, q: int) -> StructuralDiagnostics:
    """Compute the common Section 5.3.1 diagnostic bundle for one graph."""

    graph_array = _benchmark_graph(graph)
    n_agents = graph_array.shape[0]
    q = _positive_q(q, n_agents)
    total_links = int(graph_array.sum())

    return StructuralDiagnostics(
        n_agents=n_agents,
        total_links=total_links,
        mean_out_degree=float(graph_array.sum(axis=1).mean()),
        in_degree_gini=in_degree_gini(graph_array),
        hub_q=q,
        hub_link_share=hub_link_share(graph_array, q),
        global_clustering=global_clustering(graph_array),
        average_path_length_lcc=average_path_length_lcc(graph_array),
        largest_component_share=largest_component_share(graph_array),
    )


def diagnose_ensemble(
    graphs: Iterable[np.ndarray],
    *,
    q: int,
) -> tuple[StructuralDiagnostics, ...]:
    """Return graph-level diagnostics for an ensemble without collapsing its distribution."""

    try:
        graph_tuple = tuple(graphs)
    except TypeError as exc:
        raise TypeError("graphs must be an iterable of adjacency matrices") from exc
    if len(graph_tuple) == 0:
        raise ValueError("graphs must contain at least one graph")

    return tuple(diagnose_graph(graph, q=q) for graph in graph_tuple)
