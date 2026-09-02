import numpy as np
import pytest

from src.experiments.refined.influence_metrics import (
    RealisedInfluencePath,
    attention_entropy,
    attention_mobility,
    attention_overlap,
    effective_number_of_sources,
    normalised_attention_entropy,
    realised_hub_influence_share,
    realised_influence_hhi,
    realised_influence_path,
    realised_influence_shares,
    structural_hub_nodes,
)
from src.model.refined.simulator import SimulationResult
from src.model.refined.state import PeriodOutputs, RefinedState
from src.topologies.refined.diagnostics import hub_link_share, in_degrees


REGULAR_GRAPH = np.array(
    [
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
    ],
    dtype=np.int8,
)

HUB_GRAPH = np.array(
    [
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ],
    dtype=np.int8,
)

DEGREE_ONE_GRAPH = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ],
    dtype=np.int8,
)


def _uniform_attention(graph: np.ndarray) -> np.ndarray:
    graph = np.asarray(graph, dtype=float)
    return graph / graph.sum(axis=1, keepdims=True)


def _concentrated_regular_attention() -> np.ndarray:
    # Each row places all mass on one feasible source. Sources are all distinct.
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )


def _shared_regular_attention() -> np.ndarray:
    # Rows 0 and 3 share source 1; the other chosen sources are distinct.
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )


def _state(attention: np.ndarray) -> RefinedState:
    n = attention.shape[0]
    return RefinedState(
        theta=0.0,
        beliefs=np.zeros(n),
        positions=np.zeros(n),
        price=0.0,
        reputation=np.zeros(n),
        attention=attention,
    )


def _output(n: int) -> PeriodOutputs:
    zeros = np.zeros(n)
    return PeriodOutputs(
        fundamental_value=0.0,
        signals=zeros,
        perceived_values=zeros,
        valuation_gaps=zeros,
        desired_actions=zeros,
        actions=zeros,
        net_order_flow=0.0,
        return_=0.0,
        profits=zeros,
        reputation_scores=np.zeros((n, n)),
    )


def _simulation(attentions: tuple[np.ndarray, ...]) -> SimulationResult:
    n = attentions[0].shape[0]
    return SimulationResult(
        states=tuple(_state(attention) for attention in attentions),
        period_outputs=tuple(_output(n) for _ in range(len(attentions) - 1)),
    )


def test_structural_hub_nodes_select_largest_indegrees():
    assert tuple(in_degrees(HUB_GRAPH)) == (4, 4, 2, 0, 0)
    assert structural_hub_nodes(HUB_GRAPH, q=2) == (0, 1)


def test_structural_hub_nodes_use_deterministic_label_tie_break():
    assert tuple(in_degrees(REGULAR_GRAPH)) == (2, 2, 2, 2)
    assert structural_hub_nodes(REGULAR_GRAPH, q=2) == (0, 1)


@pytest.mark.parametrize(
    ("q", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (6, ValueError),
    ],
)
def test_structural_hub_nodes_reject_invalid_q(q, error):
    with pytest.raises(error):
        structural_hub_nodes(HUB_GRAPH, q=q)


def test_structural_hub_nodes_reject_self_links():
    graph = REGULAR_GRAPH.copy()
    graph[0, 0] = 1
    with pytest.raises(ValueError, match="zero self-links"):
        structural_hub_nodes(graph, q=2)


def test_uniform_attention_entropy_is_log_degree():
    attention = _uniform_attention(REGULAR_GRAPH)
    np.testing.assert_allclose(
        attention_entropy(attention, REGULAR_GRAPH),
        np.full(4, np.log(2.0)),
        atol=1e-12,
    )


def test_uniform_normalised_attention_entropy_is_one():
    attention = _uniform_attention(REGULAR_GRAPH)
    np.testing.assert_allclose(
        normalised_attention_entropy(attention, REGULAR_GRAPH),
        np.ones(4),
        atol=1e-12,
    )


def test_uniform_effective_number_of_sources_equals_degree():
    attention = _uniform_attention(REGULAR_GRAPH)
    np.testing.assert_allclose(
        effective_number_of_sources(attention, REGULAR_GRAPH),
        np.full(4, 2.0),
        atol=1e-12,
    )


def test_one_hot_attention_has_zero_entropy_and_one_effective_source():
    attention = _concentrated_regular_attention()
    np.testing.assert_allclose(attention_entropy(attention, REGULAR_GRAPH), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        normalised_attention_entropy(attention, REGULAR_GRAPH),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        effective_number_of_sources(attention, REGULAR_GRAPH),
        1.0,
        atol=1e-12,
    )


def test_normalised_entropy_rejects_degree_one_graph():
    with pytest.raises(ValueError, match="out-degree to exceed one"):
        normalised_attention_entropy(
            _uniform_attention(DEGREE_ONE_GRAPH),
            DEGREE_ONE_GRAPH,
        )


def test_entropy_rejects_attention_outside_graph_support():
    attention = _uniform_attention(REGULAR_GRAPH)
    attention[0, 0] = 0.1
    attention[0, 1] -= 0.1
    with pytest.raises(ValueError, match="outside the feasible graph"):
        attention_entropy(attention, REGULAR_GRAPH)


def test_entropy_does_not_mutate_attention():
    attention = _uniform_attention(REGULAR_GRAPH)
    before = attention.copy()
    attention_entropy(attention, REGULAR_GRAPH)
    np.testing.assert_array_equal(attention, before)


def test_uniform_fixed_out_degree_influence_shares_equal_indegree_share():
    attention = _uniform_attention(HUB_GRAPH)
    expected = in_degrees(HUB_GRAPH) / HUB_GRAPH.sum()
    np.testing.assert_allclose(
        realised_influence_shares(attention, HUB_GRAPH),
        expected,
        atol=1e-12,
    )


def test_realised_influence_shares_sum_to_one():
    shares = realised_influence_shares(_uniform_attention(HUB_GRAPH), HUB_GRAPH)
    assert shares.sum() == pytest.approx(1.0)


def test_uniform_regular_influence_hhi_is_one_over_n():
    assert realised_influence_hhi(
        _uniform_attention(REGULAR_GRAPH),
        REGULAR_GRAPH,
    ) == pytest.approx(0.25)


def test_shared_one_hot_attention_raises_global_influence_hhi():
    # Source shares are [0, 1/2, 1/4, 1/4].
    assert realised_influence_hhi(
        _shared_regular_attention(),
        REGULAR_GRAPH,
    ) == pytest.approx(0.375)


def test_uniform_initial_realised_hub_share_equals_structural_hub_share():
    attention = _uniform_attention(HUB_GRAPH)
    assert realised_hub_influence_share(
        attention,
        HUB_GRAPH,
        q=2,
    ) == pytest.approx(hub_link_share(HUB_GRAPH, 2))


def test_uniform_regular_attention_overlap_has_closed_form_value():
    # Four of the six unordered row pairs share exactly one source at weight 1/2.
    # Hence average inner product = 4*(1/4)/6 = 1/6.
    assert attention_overlap(
        _uniform_attention(REGULAR_GRAPH),
        REGULAR_GRAPH,
    ) == pytest.approx(1.0 / 6.0)


def test_disjoint_one_hot_attention_overlap_is_zero():
    assert attention_overlap(
        _concentrated_regular_attention(),
        REGULAR_GRAPH,
    ) == pytest.approx(0.0)


def test_common_one_hot_source_creates_positive_overlap():
    # Exactly one of six unordered pairs has identical one-hot attention.
    assert attention_overlap(
        _shared_regular_attention(),
        REGULAR_GRAPH,
    ) == pytest.approx(1.0 / 6.0)


def test_attention_mobility_is_zero_when_attention_does_not_change():
    attention = _uniform_attention(REGULAR_GRAPH)
    assert attention_mobility(attention, attention, REGULAR_GRAPH) == pytest.approx(0.0)


def test_attention_mobility_uses_rms_row_change_normalisation():
    uniform = _uniform_attention(REGULAR_GRAPH)
    concentrated = _concentrated_regular_attention()
    # Each row changes from (1/2,1/2) to one-hot on one supported source:
    # squared row distance = 1/2, so RMS row distance = sqrt(1/2).
    assert attention_mobility(
        concentrated,
        uniform,
        REGULAR_GRAPH,
    ) == pytest.approx(np.sqrt(0.5))


def test_attention_mobility_is_symmetric_as_euclidean_distance():
    uniform = _uniform_attention(REGULAR_GRAPH)
    concentrated = _concentrated_regular_attention()
    assert attention_mobility(
        concentrated,
        uniform,
        REGULAR_GRAPH,
    ) == pytest.approx(
        attention_mobility(uniform, concentrated, REGULAR_GRAPH)
    )


def test_realised_influence_path_uses_period_states_not_initial_state_as_period_one():
    w0 = _uniform_attention(REGULAR_GRAPH)
    w1 = _concentrated_regular_attention()
    result = _simulation((w0, w1))
    path = realised_influence_path(result, REGULAR_GRAPH, q=2)
    assert isinstance(path, RealisedInfluencePath)
    assert len(path.points) == 1
    np.testing.assert_allclose(path.points[0].normalised_entropies, 0.0, atol=1e-12)


def test_realised_influence_path_first_mobility_is_w0_to_w1():
    w0 = _uniform_attention(REGULAR_GRAPH)
    w1 = _concentrated_regular_attention()
    result = _simulation((w0, w1))
    path = realised_influence_path(result, REGULAR_GRAPH, q=2)
    assert path.points[0].attention_mobility == pytest.approx(np.sqrt(0.5))


def test_realised_influence_path_second_mobility_zero_when_w_is_unchanged():
    w0 = _uniform_attention(REGULAR_GRAPH)
    w1 = _concentrated_regular_attention()
    result = _simulation((w0, w1, w1))
    path = realised_influence_path(result, REGULAR_GRAPH, q=2)
    assert tuple(point.period for point in path.points) == (1, 2)
    assert path.points[1].attention_mobility == pytest.approx(0.0)


def test_realised_influence_path_records_structural_hubs_once():
    w = _uniform_attention(HUB_GRAPH)
    result = _simulation((w, w))
    path = realised_influence_path(result, HUB_GRAPH, q=2)
    assert path.hub_q == 2
    assert path.structural_hubs == (0, 1)


def test_realised_influence_path_network_averages_match_agent_arrays():
    w0 = _uniform_attention(REGULAR_GRAPH)
    w1 = _shared_regular_attention()
    point = realised_influence_path(
        _simulation((w0, w1)),
        REGULAR_GRAPH,
        q=2,
    ).points[0]
    assert point.mean_normalised_entropy == pytest.approx(np.mean(point.normalised_entropies))
    assert point.mean_effective_sources == pytest.approx(np.mean(point.effective_sources))


def test_realised_influence_path_hhi_matches_saved_source_shares():
    w0 = _uniform_attention(REGULAR_GRAPH)
    w1 = _shared_regular_attention()
    point = realised_influence_path(
        _simulation((w0, w1)),
        REGULAR_GRAPH,
        q=2,
    ).points[0]
    assert point.influence_hhi == pytest.approx(np.sum(point.source_influence_shares**2))


def test_realised_influence_path_rejects_graph_state_dimension_mismatch():
    result = _simulation(
        (_uniform_attention(REGULAR_GRAPH), _uniform_attention(REGULAR_GRAPH))
    )
    with pytest.raises(ValueError, match="dimensions must agree"):
        realised_influence_path(result, HUB_GRAPH, q=2)


def test_realised_influence_path_rejects_degree_one_entropy_case():
    w = _uniform_attention(DEGREE_ONE_GRAPH)
    result = _simulation((w, w))
    with pytest.raises(ValueError, match="out-degree to exceed one"):
        realised_influence_path(result, DEGREE_ONE_GRAPH, q=1)


def test_realised_influence_path_rejects_non_simulation_result():
    with pytest.raises(TypeError, match="SimulationResult"):
        realised_influence_path(object(), REGULAR_GRAPH, q=2)
