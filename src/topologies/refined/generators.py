"""Report-faithful directed benchmark topology generators for Section 5.3.

The refined first-stage experiment treats ``G`` as a fixed feasible-information
support, distinct from effective attention ``W``.  All three benchmark classes
constructed here are directed simple graphs with exactly ``K`` outgoing links
per agent, no self-links, and no duplicate directed edges.

Only ``graph_seed`` controls graph randomness.  Shock, initial-state, and type
assignment randomness must remain outside this module.
"""

from __future__ import annotations

import numpy as np

from src.model.refined.state import validate_graph_support


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _nonnegative_seed(graph_seed: int) -> int:
    if isinstance(graph_seed, bool) or not isinstance(graph_seed, (int, np.integer)):
        raise TypeError("graph_seed must be an integer")
    graph_seed = int(graph_seed)
    if graph_seed < 0:
        raise ValueError("graph_seed must be non-negative")
    return graph_seed


def _common_inputs(n_agents: int, k: int, graph_seed: int) -> tuple[int, int, int]:
    n_agents = _positive_integer("n_agents", n_agents)
    k = _positive_integer("k", k)
    graph_seed = _nonnegative_seed(graph_seed)

    if n_agents < 2:
        raise ValueError("n_agents must be at least two")
    if k > n_agents - 1:
        raise ValueError("k must satisfy 1 <= k <= n_agents - 1")
    return n_agents, k, graph_seed


def _probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return value


def _positive_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _validate_benchmark_graph(graph: np.ndarray, k: int) -> np.ndarray:
    """Enforce the common Section 5.3 benchmark restrictions, Equation (191)."""

    graph_array = validate_graph_support(graph)
    if np.any(np.diag(graph_array) != 0):
        raise ValueError("benchmark graphs must exclude self-links")
    if not np.all(graph_array.sum(axis=1) == k):
        raise ValueError("every benchmark graph row must contain exactly k links")
    return graph_array


def generate_random_fixed_out_degree(
    *,
    n_agents: int,
    k: int,
    graph_seed: int,
) -> np.ndarray:
    """Generate the directed fixed-out-degree Random benchmark, Eqs. (192)-(196).

    For each source agent ``i``, exactly ``k`` distinct targets are sampled
    uniformly without replacement from the remaining ``n_agents - 1`` nodes.
    Row draws are independent conditional on the common graph seed.
    """

    n_agents, k, graph_seed = _common_inputs(n_agents, k, graph_seed)
    rng = np.random.default_rng(graph_seed)
    graph = np.zeros((n_agents, n_agents), dtype=np.int8)
    nodes = np.arange(n_agents, dtype=int)

    for i in range(n_agents):
        eligible = nodes[nodes != i]
        targets = rng.choice(eligible, size=k, replace=False)
        graph[i, targets] = 1

    return _validate_benchmark_graph(graph, k)


def generate_small_world(
    *,
    n_agents: int,
    k: int,
    p_sw: float,
    graph_seed: int,
) -> np.ndarray:
    """Generate the directed Small-World benchmark, Equation (197).

    The initial graph is a directed ring lattice with ``k/2`` nearest feasible
    sources on either side of each agent.  ``k`` must therefore be even.  Each
    outgoing lattice edge is independently selected for rewiring with
    probability ``p_sw``.  Its source is retained and its replacement target
    is sampled uniformly from nodes that preserve the no-self-link and
    no-duplicate restrictions.  Every row consequently retains exactly
    ``k`` outgoing links.
    """

    n_agents, k, graph_seed = _common_inputs(n_agents, k, graph_seed)
    p_sw = _probability("p_sw", p_sw)
    if k % 2 != 0:
        raise ValueError("small-world benchmark requires even k")
    if p_sw > 0.0 and k == n_agents - 1:
        raise ValueError("rewiring requires at least one non-neighbour target")

    rng = np.random.default_rng(graph_seed)
    graph = np.zeros((n_agents, n_agents), dtype=np.int8)
    half = k // 2

    lattice_targets: list[tuple[int, ...]] = []
    for i in range(n_agents):
        targets = tuple(
            [(i + offset) % n_agents for offset in range(1, half + 1)]
            + [(i - offset) % n_agents for offset in range(1, half + 1)]
        )
        if len(set(targets)) != k or i in targets:
            raise ValueError("n_agents and k do not admit the requested ring lattice")
        graph[i, list(targets)] = 1
        lattice_targets.append(targets)

    if p_sw == 0.0:
        return _validate_benchmark_graph(graph, k)

    nodes = np.arange(n_agents, dtype=int)
    for i, original_targets in enumerate(lattice_targets):
        for old_target in original_targets:
            if rng.random() >= p_sw:
                continue

            graph[i, old_target] = 0
            eligible_mask = (nodes != i) & (graph[i] == 0) & (nodes != old_target)
            eligible = nodes[eligible_mask]
            if eligible.size == 0:
                graph[i, old_target] = 1
                raise ValueError("no eligible replacement target is available for rewiring")

            new_target = int(rng.choice(eligible))
            graph[i, new_target] = 1

    return _validate_benchmark_graph(graph, k)


def generate_hub_dominated(
    *,
    n_agents: int,
    k: int,
    a0: float,
    graph_seed: int,
) -> np.ndarray:
    """Generate the preferential-attachment hub-dominated benchmark, Eqs. (198)-(201).

    There are exactly ``n_agents * k`` directed edge slots.  Source labels are
    randomly ordered subject to each source appearing exactly ``k`` times.
    When a slot belonging to source ``i`` is allocated, an eligible target
    ``j`` is sampled with probability proportional to

    ``in_degree[j] + a0``.

    The positive initial-attractiveness offset ``a0`` ensures that zero-degree
    nodes retain positive attachment probability.  After the graph is formed,
    an independent child RNG stream randomly relabels all nodes as required by
    Equation (212), preventing arbitrary numerical labels from being attached
    mechanically to structural hub positions.

    The report uses the label Scale-Free/SF for continuity, but this finite
    construction is interpreted as hub-dominated unless a separate degree-
    distribution diagnostic supports a stronger claim.
    """

    n_agents, k, graph_seed = _common_inputs(n_agents, k, graph_seed)
    a0 = _positive_scalar("a0", a0)

    root_sequence = np.random.SeedSequence(graph_seed)
    formation_sequence, relabelling_sequence = root_sequence.spawn(2)
    rng = np.random.default_rng(formation_sequence)
    relabelling_rng = np.random.default_rng(relabelling_sequence)

    graph = np.zeros((n_agents, n_agents), dtype=np.int8)
    in_degree = np.zeros(n_agents, dtype=np.int64)
    source_slots = np.repeat(np.arange(n_agents, dtype=int), k)
    rng.shuffle(source_slots)
    nodes = np.arange(n_agents, dtype=int)

    for source in source_slots:
        eligible_mask = (nodes != source) & (graph[source] == 0)
        eligible = nodes[eligible_mask]
        if eligible.size == 0:
            raise ValueError("no eligible target remains for an outgoing edge slot")

        weights = in_degree[eligible].astype(float) + a0
        probabilities = weights / np.sum(weights)
        target = int(rng.choice(eligible, p=probabilities))
        graph[source, target] = 1
        in_degree[target] += 1

    permutation = relabelling_rng.permutation(n_agents)
    relabelled = np.zeros_like(graph)
    relabelled[np.ix_(permutation, permutation)] = graph

    return _validate_benchmark_graph(relabelled, k)
