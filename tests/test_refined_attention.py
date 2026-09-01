"""Tests for graph-supported attention, Equations (41) and (57)-(60)."""

import numpy as np
import pytest

from src.model.refined import (
    local_reputation_statistics,
    standardised_reputation_scores,
    uniform_attention_from_graph,
    update_attention,
    validate_attention,
)


def example_graph():
    return np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )


def test_uniform_attention_from_graph_matches_neutral_rule():
    attention = uniform_attention_from_graph(example_graph())
    expected = np.array(
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    np.testing.assert_allclose(attention, expected)


def test_uniform_attention_from_graph_reduces_to_one_over_k_for_fixed_out_degree():
    graph = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    attention = uniform_attention_from_graph(graph)
    np.testing.assert_allclose(attention[graph == 1], 0.5)
    np.testing.assert_array_equal(attention[graph == 0], np.zeros(np.sum(graph == 0)))


def test_uniform_attention_from_graph_rejects_row_without_feasible_source():
    invalid = np.array([[0, 0], [1, 0]])
    with pytest.raises(ValueError, match="at least one feasible"):
        uniform_attention_from_graph(invalid)


def test_local_reputation_statistics_match_equations_57_and_58():
    reputation = np.array([1.0, 3.0, 5.0])
    means, dispersions = local_reputation_statistics(
        reputation,
        example_graph(),
        sigma_0=2.0,
    )

    assert np.allclose(means, np.array([4.0, 1.0, 2.0]))
    assert np.allclose(dispersions, np.array([np.sqrt(5.0), 2.0, np.sqrt(5.0)]))


def test_standardised_reputation_scores_match_equation_59():
    reputation = np.array([1.0, 3.0, 5.0])
    scores = standardised_reputation_scores(
        reputation,
        example_graph(),
        sigma_0=2.0,
    )
    scale = 1.0 / np.sqrt(5.0)
    expected = np.array(
        [
            [0.0, -scale, scale],
            [0.0, 0.0, 0.0],
            [-scale, scale, 0.0],
        ]
    )
    assert np.allclose(scores, expected)


def test_beta_zero_reduces_to_uniform_graph_supported_attention():
    attention, _ = update_attention(
        np.array([1.0, 3.0, 5.0]),
        example_graph(),
        beta=0.0,
        sigma_0=0.1,
    )
    expected = uniform_attention_from_graph(example_graph())
    assert np.allclose(attention, expected)


def test_positive_beta_matches_manual_softmax_on_feasible_neighbours():
    graph = example_graph()
    reputation = np.array([1.0, 3.0, 5.0])
    beta = 2.0
    sigma_0 = 2.0
    attention, scores = update_attention(reputation, graph, beta, sigma_0)

    row0_logits = beta * scores[0, [1, 2]]
    row0_expected = np.exp(row0_logits - np.max(row0_logits))
    row0_expected /= row0_expected.sum()

    assert np.allclose(attention[0, [1, 2]], row0_expected)
    assert attention[0, 0] == 0.0


def test_equal_local_reputations_remain_uniform_for_positive_beta():
    graph = example_graph()
    attention, scores = update_attention(
        np.ones(3),
        graph,
        beta=8.0,
        sigma_0=1e-3,
    )

    assert np.allclose(scores, np.zeros((3, 3)))
    assert np.allclose(attention, uniform_attention_from_graph(graph))


def test_attention_is_row_stochastic_and_exactly_zero_off_graph_support():
    graph = example_graph()
    attention, _ = update_attention(
        np.array([-3.0, 2.0, 7.0]),
        graph,
        beta=3.0,
        sigma_0=0.2,
    )

    checked = validate_attention(attention, graph)
    assert np.allclose(checked.sum(axis=1), 1.0)
    assert np.array_equal(checked[graph == 0], np.zeros(np.sum(graph == 0)))


def test_larger_beta_increases_weight_on_better_reputation_source():
    graph = example_graph()
    reputation = np.array([1.0, 3.0, 5.0])
    low_beta, _ = update_attention(reputation, graph, beta=0.5, sigma_0=0.1)
    high_beta, _ = update_attention(reputation, graph, beta=4.0, sigma_0=0.1)

    assert high_beta[0, 2] > low_beta[0, 2]
    assert high_beta[0, 1] < low_beta[0, 1]


def test_reputation_translation_does_not_change_scores_or_attention():
    graph = example_graph()
    reputation = np.array([1.0, 3.0, 5.0])
    attention_a, scores_a = update_attention(reputation, graph, beta=2.0, sigma_0=0.3)
    attention_b, scores_b = update_attention(reputation + 100.0, graph, beta=2.0, sigma_0=0.3)

    assert np.allclose(scores_a, scores_b)
    assert np.allclose(attention_a, attention_b)


def test_single_feasible_neighbour_always_receives_unit_weight():
    graph = np.array([[0, 1], [1, 0]])
    attention, _ = update_attention(
        np.array([-100.0, 100.0]),
        graph,
        beta=100.0,
        sigma_0=1e-6,
    )
    assert np.array_equal(attention, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_max_shifted_softmax_remains_finite_for_large_beta():
    attention, _ = update_attention(
        np.array([0.0, -1.0, 1.0]),
        example_graph(),
        beta=1000.0,
        sigma_0=0.1,
    )
    assert np.all(np.isfinite(attention))
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert attention[0, 2] > 0.999


@pytest.mark.parametrize(
    ("beta", "sigma_0", "reputation"),
    [
        (-0.1, 0.1, np.array([1.0, 2.0, 3.0])),
        (np.inf, 0.1, np.array([1.0, 2.0, 3.0])),
        (1.0, 0.0, np.array([1.0, 2.0, 3.0])),
        (1.0, -0.1, np.array([1.0, 2.0, 3.0])),
        (1.0, np.inf, np.array([1.0, 2.0, 3.0])),
        (1.0, 0.1, np.array([1.0, np.nan, 3.0])),
    ],
)
def test_update_attention_rejects_invalid_inputs(beta, sigma_0, reputation):
    with pytest.raises(ValueError):
        update_attention(reputation, example_graph(), beta=beta, sigma_0=sigma_0)
