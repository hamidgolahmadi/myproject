"""Tests for Section 5.3.1 structural graph diagnostics, Eqs. (203)-(211)."""

from __future__ import annotations

import numpy as np
import pytest

from src.topologies.refined import (
    average_path_length_lcc,
    diagnose_ensemble,
    diagnose_graph,
    global_clustering,
    hub_link_share,
    in_degree_gini,
    in_degrees,
    largest_component_share,
    symmetrised_support,
)


def star_graph() -> np.ndarray:
    # Directed edges: 0->1, 1->0, 2->0, 3->0.
    # In-degrees are therefore [3, 1, 0, 0], while G^u is a four-node star.
    return np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int8,
    )


def directed_cycle() -> np.ndarray:
    return np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
        dtype=np.int8,
    )


def disconnected_pairs() -> np.ndarray:
    return np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.int8,
    )


def test_in_degrees_are_directed_column_sums() -> None:
    np.testing.assert_array_equal(in_degrees(star_graph()), np.array([3, 1, 0, 0]))


def test_in_degree_gini_matches_equation_203() -> None:
    assert in_degree_gini(star_graph()) == pytest.approx(0.625)


def test_hub_link_share_matches_equation_206() -> None:
    assert hub_link_share(star_graph(), q=1) == pytest.approx(0.75)
    assert hub_link_share(star_graph(), q=2) == pytest.approx(1.0)


def test_symmetrised_support_matches_equation_208() -> None:
    expected = np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(symmetrised_support(star_graph()), expected)


def test_complete_support_has_unit_global_clustering() -> None:
    graph = np.ones((4, 4), dtype=np.int8) - np.eye(4, dtype=np.int8)
    assert global_clustering(graph) == pytest.approx(1.0)


def test_four_cycle_has_zero_global_clustering() -> None:
    assert global_clustering(directed_cycle()) == pytest.approx(0.0)


def test_four_cycle_average_path_length_is_four_thirds() -> None:
    assert average_path_length_lcc(directed_cycle()) == pytest.approx(4.0 / 3.0)


def test_largest_component_share_uses_symmetrised_component_structure() -> None:
    assert largest_component_share(disconnected_pairs()) == pytest.approx(0.5)


def test_average_path_length_is_computed_only_inside_largest_component() -> None:
    assert average_path_length_lcc(disconnected_pairs()) == pytest.approx(1.0)


def test_diagnose_graph_returns_exact_star_bundle() -> None:
    diagnostics = diagnose_graph(star_graph(), q=1)

    assert diagnostics.n_agents == 4
    assert diagnostics.total_links == 4
    assert diagnostics.mean_out_degree == pytest.approx(1.0)
    assert diagnostics.in_degree_gini == pytest.approx(0.625)
    assert diagnostics.hub_q == 1
    assert diagnostics.hub_link_share == pytest.approx(0.75)
    assert diagnostics.global_clustering == pytest.approx(0.0)
    assert diagnostics.average_path_length_lcc == pytest.approx(1.5)
    assert diagnostics.largest_component_share == pytest.approx(1.0)


def test_structural_diagnostics_are_invariant_to_node_relabelling() -> None:
    graph = star_graph()
    permutation = np.array([2, 0, 3, 1])
    relabelled = graph[np.ix_(permutation, permutation)]

    original = diagnose_graph(graph, q=1)
    permuted = diagnose_graph(relabelled, q=1)

    assert original.total_links == permuted.total_links
    assert original.mean_out_degree == pytest.approx(permuted.mean_out_degree)
    assert original.in_degree_gini == pytest.approx(permuted.in_degree_gini)
    assert original.hub_link_share == pytest.approx(permuted.hub_link_share)
    assert original.global_clustering == pytest.approx(permuted.global_clustering)
    assert original.average_path_length_lcc == pytest.approx(
        permuted.average_path_length_lcc
    )
    assert original.largest_component_share == pytest.approx(
        permuted.largest_component_share
    )


def test_diagnose_ensemble_preserves_graph_level_distribution_and_order() -> None:
    diagnostics = diagnose_ensemble([star_graph(), directed_cycle()], q=1)
    assert len(diagnostics) == 2
    assert diagnostics[0].in_degree_gini == pytest.approx(0.625)
    assert diagnostics[1].in_degree_gini == pytest.approx(0.0)


def test_diagnose_ensemble_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        diagnose_ensemble([], q=1)


@pytest.mark.parametrize(
    ("q", "error_type"),
    [
        (0, ValueError),
        (5, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_hub_q_is_validated(q, error_type) -> None:
    with pytest.raises(error_type):
        hub_link_share(star_graph(), q=q)


def test_structural_diagnostics_reject_self_links() -> None:
    graph = np.eye(2, dtype=np.int8)
    with pytest.raises(ValueError, match="zero self-links"):
        diagnose_graph(graph, q=1)


def test_structural_diagnostics_preserve_core_graph_validation() -> None:
    graph = np.array([[0, 1], [0, 0]], dtype=np.int8)
    with pytest.raises(ValueError):
        diagnose_graph(graph, q=1)


def test_complete_directed_support_has_uniform_degree_and_expected_hub_share() -> None:
    graph = np.ones((5, 5), dtype=np.int8) - np.eye(5, dtype=np.int8)
    diagnostics = diagnose_graph(graph, q=2)

    assert diagnostics.in_degree_gini == pytest.approx(0.0)
    assert diagnostics.hub_link_share == pytest.approx(2.0 / 5.0)
    assert diagnostics.global_clustering == pytest.approx(1.0)
    assert diagnostics.average_path_length_lcc == pytest.approx(1.0)
    assert diagnostics.largest_component_share == pytest.approx(1.0)
