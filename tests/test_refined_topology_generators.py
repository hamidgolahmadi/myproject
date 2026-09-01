"""Tests for the report-defined refined benchmark graph generators."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.refined.state import validate_graph_support
from src.topologies.refined import (
    generate_hub_dominated,
    generate_random_fixed_out_degree,
    generate_small_world,
)


def _assert_benchmark_invariants(graph: np.ndarray, n_agents: int, k: int) -> None:
    assert graph.shape == (n_agents, n_agents)
    assert graph.dtype == np.int8
    assert np.all(np.isin(graph, (0, 1)))
    assert np.all(np.diag(graph) == 0)
    np.testing.assert_array_equal(graph.sum(axis=1), np.full(n_agents, k))
    assert int(graph.sum()) == n_agents * k
    np.testing.assert_array_equal(validate_graph_support(graph), graph)


def test_random_fixed_out_degree_satisfies_section_53_invariants() -> None:
    graph = generate_random_fixed_out_degree(n_agents=30, k=6, graph_seed=101)
    _assert_benchmark_invariants(graph, 30, 6)


def test_random_fixed_out_degree_is_reproducible() -> None:
    first = generate_random_fixed_out_degree(n_agents=25, k=4, graph_seed=202)
    second = generate_random_fixed_out_degree(n_agents=25, k=4, graph_seed=202)
    np.testing.assert_array_equal(first, second)


def test_random_fixed_out_degree_changes_with_graph_seed() -> None:
    first = generate_random_fixed_out_degree(n_agents=25, k=4, graph_seed=1)
    second = generate_random_fixed_out_degree(n_agents=25, k=4, graph_seed=2)
    assert not np.array_equal(first, second)


def test_random_fixed_out_degree_allows_complete_directed_support() -> None:
    graph = generate_random_fixed_out_degree(n_agents=7, k=6, graph_seed=303)
    expected = np.ones((7, 7), dtype=np.int8) - np.eye(7, dtype=np.int8)
    np.testing.assert_array_equal(graph, expected)


def test_small_world_zero_rewiring_is_exact_directed_ring_lattice() -> None:
    graph = generate_small_world(n_agents=8, k=4, p_sw=0.0, graph_seed=404)

    expected = np.zeros((8, 8), dtype=np.int8)
    for i in range(8):
        targets = [
            (i + 1) % 8,
            (i + 2) % 8,
            (i - 1) % 8,
            (i - 2) % 8,
        ]
        expected[i, targets] = 1

    np.testing.assert_array_equal(graph, expected)
    _assert_benchmark_invariants(graph, 8, 4)


def test_small_world_rewiring_preserves_link_budget_and_simple_graph() -> None:
    graph = generate_small_world(n_agents=30, k=6, p_sw=0.25, graph_seed=505)
    _assert_benchmark_invariants(graph, 30, 6)


def test_small_world_is_reproducible() -> None:
    first = generate_small_world(n_agents=30, k=6, p_sw=0.3, graph_seed=606)
    second = generate_small_world(n_agents=30, k=6, p_sw=0.3, graph_seed=606)
    np.testing.assert_array_equal(first, second)


def test_small_world_full_rewiring_changes_the_lattice_for_typical_case() -> None:
    lattice = generate_small_world(n_agents=20, k=4, p_sw=0.0, graph_seed=707)
    rewired = generate_small_world(n_agents=20, k=4, p_sw=1.0, graph_seed=707)
    assert not np.array_equal(lattice, rewired)
    _assert_benchmark_invariants(rewired, 20, 4)


def test_small_world_rejects_odd_k() -> None:
    with pytest.raises(ValueError, match="even k"):
        generate_small_world(n_agents=20, k=5, p_sw=0.1, graph_seed=808)


@pytest.mark.parametrize("p_sw", [-0.01, 1.01, np.inf])
def test_small_world_rejects_invalid_rewiring_probability(p_sw: float) -> None:
    with pytest.raises(ValueError, match="p_sw"):
        generate_small_world(n_agents=20, k=4, p_sw=p_sw, graph_seed=909)


def test_small_world_rejects_rewiring_when_every_other_node_is_already_a_neighbour() -> None:
    with pytest.raises(ValueError, match="non-neighbour"):
        generate_small_world(n_agents=7, k=6, p_sw=0.2, graph_seed=1001)


def test_hub_dominated_satisfies_section_53_invariants() -> None:
    graph = generate_hub_dominated(n_agents=40, k=6, a0=1.0, graph_seed=1101)
    _assert_benchmark_invariants(graph, 40, 6)


def test_hub_dominated_is_reproducible() -> None:
    first = generate_hub_dominated(n_agents=35, k=5, a0=1.0, graph_seed=1201)
    second = generate_hub_dominated(n_agents=35, k=5, a0=1.0, graph_seed=1201)
    np.testing.assert_array_equal(first, second)


def test_hub_dominated_changes_with_graph_seed() -> None:
    first = generate_hub_dominated(n_agents=35, k=5, a0=1.0, graph_seed=13)
    second = generate_hub_dominated(n_agents=35, k=5, a0=1.0, graph_seed=14)
    assert not np.array_equal(first, second)


def test_hub_dominated_allows_complete_directed_support() -> None:
    graph = generate_hub_dominated(n_agents=6, k=5, a0=0.5, graph_seed=1301)
    expected = np.ones((6, 6), dtype=np.int8) - np.eye(6, dtype=np.int8)
    np.testing.assert_array_equal(graph, expected)


@pytest.mark.parametrize("a0", [0.0, -1.0, np.inf])
def test_hub_dominated_rejects_invalid_initial_attractiveness(a0: float) -> None:
    with pytest.raises(ValueError, match="a0"):
        generate_hub_dominated(n_agents=20, k=4, a0=a0, graph_seed=1401)


@pytest.mark.parametrize(
    ("generator", "kwargs", "error_type", "match"),
    [
        (
            generate_random_fixed_out_degree,
            {"n_agents": 1, "k": 1, "graph_seed": 1},
            ValueError,
            "at least two",
        ),
        (
            generate_random_fixed_out_degree,
            {"n_agents": 5, "k": 5, "graph_seed": 1},
            ValueError,
            "n_agents - 1",
        ),
        (
            generate_random_fixed_out_degree,
            {"n_agents": 5, "k": 0, "graph_seed": 1},
            ValueError,
            "strictly positive",
        ),
        (
            generate_random_fixed_out_degree,
            {"n_agents": 5, "k": 2, "graph_seed": -1},
            ValueError,
            "non-negative",
        ),
        (
            generate_random_fixed_out_degree,
            {"n_agents": 5.0, "k": 2, "graph_seed": 1},
            TypeError,
            "integer",
        ),
        (
            generate_random_fixed_out_degree,
            {"n_agents": 5, "k": 2.0, "graph_seed": 1},
            TypeError,
            "integer",
        ),
        (
            generate_random_fixed_out_degree,
            {"n_agents": 5, "k": 2, "graph_seed": True},
            TypeError,
            "integer",
        ),
    ],
)
def test_common_topology_inputs_are_validated(
    generator,
    kwargs: dict[str, object],
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        generator(**kwargs)


def test_generators_use_only_graph_specific_arguments() -> None:
    """Public graph APIs deliberately expose no shock or simulation seed arguments."""

    import inspect

    for generator in (
        generate_random_fixed_out_degree,
        generate_small_world,
        generate_hub_dominated,
    ):
        parameter_names = set(inspect.signature(generator).parameters)
        assert "graph_seed" in parameter_names
        assert "shock_seed" not in parameter_names
        assert "initial_state_seed" not in parameter_names
        assert "type_assignment_seed" not in parameter_names
