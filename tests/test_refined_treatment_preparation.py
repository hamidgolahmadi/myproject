"""Tests for matched refined topology-treatment preparation."""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.refined import (
    NonNetworkInitialConditions,
    TopologySpecification,
    prepare_paired_replication,
    prepare_paired_treatments,
)
from src.model.refined import RefinedParameters


def parameters() -> RefinedParameters:
    return RefinedParameters(
        rho_theta=0.8,
        sigma_theta=0.2,
        v_bar=100.0,
        psi=1.0,
        sigma_s=0.3,
        sigma_b=0.1,
        alpha=0.25,
        kappa=0.7,
        x_bar=2.0,
        chi=0.15,
        lambda_price=0.05,
        sigma_p=0.1,
        gamma_R=0.8,
        beta=2.0,
        sigma_0=0.1,
    )


def specifications(k: int = 2) -> tuple[TopologySpecification, ...]:
    return (
        TopologySpecification(topology_label="R", kind="random", k=k),
        TopologySpecification(topology_label="SW", kind="small_world", k=k, p_sw=0.25),
        TopologySpecification(topology_label="SF", kind="hub_dominated", k=k, a0=1.0),
    )


def initial_conditions(n_agents: int = 8, *, position: float = 0.0) -> NonNetworkInitialConditions:
    return NonNetworkInitialConditions(
        theta=0.2,
        beliefs=np.linspace(-0.3, 0.4, n_agents),
        positions=np.full(n_agents, position),
        price=100.1,
        reputation=np.zeros(n_agents),
    )


def plan(n_agents: int = 8):
    return prepare_paired_replication(
        experiment_seed=12345,
        replication_id=7,
        topology_labels=("R", "SW", "SF"),
        n_periods=5,
        n_agents=n_agents,
        parameters=parameters(),
    )


@pytest.mark.parametrize(
    "specification",
    [
        TopologySpecification(topology_label="R", kind="random", k=2),
        TopologySpecification(topology_label="SW", kind="small_world", k=2, p_sw=0.2),
        TopologySpecification(topology_label="SF", kind="hub_dominated", k=2, a0=1.0),
    ],
)
def test_topology_specification_accepts_valid_report_benchmarks(specification):
    assert specification.k == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topology_label": "", "kind": "random", "k": 2},
        {"topology_label": "R", "kind": "unknown", "k": 2},
        {"topology_label": "R", "kind": "random", "k": 0},
        {"topology_label": "R", "kind": "random", "k": 2, "p_sw": 0.2},
        {"topology_label": "SW", "kind": "small_world", "k": 2},
        {"topology_label": "SW", "kind": "small_world", "k": 3, "p_sw": 0.2},
        {"topology_label": "SF", "kind": "hub_dominated", "k": 2},
        {"topology_label": "SF", "kind": "hub_dominated", "k": 2, "a0": 0.0},
    ],
)
def test_topology_specification_rejects_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        TopologySpecification(**kwargs)


def test_nonnetwork_initial_conditions_copy_inputs_and_report_dimension():
    beliefs = np.array([0.1, 0.2, 0.3])
    positions = np.zeros(3)
    reputation = np.array([1.0, 2.0, 3.0])
    initial = NonNetworkInitialConditions(
        theta=0.4,
        beliefs=beliefs,
        positions=positions,
        price=99.0,
        reputation=reputation,
    )

    beliefs[0] = 999.0
    positions[0] = 999.0
    reputation[0] = 999.0

    assert initial.n_agents == 3
    assert initial.beliefs[0] == 0.1
    assert initial.positions[0] == 0.0
    assert initial.reputation[0] == 1.0


def test_nonnetwork_initial_conditions_reject_dimension_mismatch():
    with pytest.raises(ValueError, match="shape"):
        NonNetworkInitialConditions(
            theta=0.0,
            beliefs=np.zeros(3),
            positions=np.zeros(2),
            price=100.0,
            reputation=np.zeros(3),
        )


def test_prepare_paired_treatments_preserves_plan_label_order():
    paired_plan = plan()
    treatments = prepare_paired_treatments(
        plan=paired_plan,
        specifications=specifications(),
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )
    assert tuple(t.topology_label for t in treatments) == paired_plan.topology_labels


def test_prepared_graphs_share_common_k_and_benchmark_invariants():
    treatments = prepare_paired_treatments(
        plan=plan(),
        specifications=specifications(),
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )

    for treatment in treatments:
        graph = treatment.graph
        assert graph.shape == (8, 8)
        assert np.all(np.diag(graph) == 0)
        np.testing.assert_array_equal(graph.sum(axis=1), np.full(8, 2))
        assert int(graph.sum()) == 16


def test_initial_attention_is_uniform_on_each_realised_graph():
    treatments = prepare_paired_treatments(
        plan=plan(),
        specifications=specifications(),
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )

    for treatment in treatments:
        graph = treatment.graph
        attention = treatment.initial_state.attention
        np.testing.assert_allclose(attention[graph == 1], 0.5)
        np.testing.assert_array_equal(
            attention[graph == 0],
            np.zeros(np.sum(graph == 0)),
        )
        np.testing.assert_allclose(attention.sum(axis=1), 1.0)


def test_nonnetwork_initial_values_are_identical_across_treatments():
    common = initial_conditions()
    treatments = prepare_paired_treatments(
        plan=plan(),
        specifications=specifications(),
        initial_conditions=common,
        parameters=parameters(),
    )

    for treatment in treatments:
        state = treatment.initial_state
        assert state.theta == common.theta
        assert state.price == common.price
        np.testing.assert_array_equal(state.beliefs, common.beliefs)
        np.testing.assert_array_equal(state.positions, common.positions)
        np.testing.assert_array_equal(state.reputation, common.reputation)


def test_treatments_reuse_the_same_realised_shock_objects():
    paired_plan = plan()
    treatments = prepare_paired_treatments(
        plan=paired_plan,
        specifications=specifications(),
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )

    for treatment in treatments:
        assert len(treatment.shock_path) == len(paired_plan.shock_path)
        for treatment_shock, plan_shock in zip(treatment.shock_path, paired_plan.shock_path):
            assert treatment_shock is plan_shock


def test_graph_seeds_match_the_paired_plan():
    paired_plan = plan()
    treatments = prepare_paired_treatments(
        plan=paired_plan,
        specifications=specifications(),
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )

    for treatment in treatments:
        assert treatment.graph_seed == paired_plan.graph_seed_for(treatment.topology_label)


def test_specification_input_order_does_not_change_prepared_treatments():
    paired_plan = plan()
    ordered = specifications()
    reversed_input = tuple(reversed(ordered))

    first = prepare_paired_treatments(
        plan=paired_plan,
        specifications=ordered,
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )
    second = prepare_paired_treatments(
        plan=paired_plan,
        specifications=reversed_input,
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )

    assert tuple(t.topology_label for t in first) == tuple(t.topology_label for t in second)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.graph, right.graph)
        np.testing.assert_array_equal(left.initial_state.attention, right.initial_state.attention)


def test_preparation_is_reproducible():
    paired_plan = plan()
    first = prepare_paired_treatments(
        plan=paired_plan,
        specifications=specifications(),
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )
    second = prepare_paired_treatments(
        plan=paired_plan,
        specifications=specifications(),
        initial_conditions=initial_conditions(),
        parameters=parameters(),
    )

    for left, right in zip(first, second):
        assert left.graph_seed == right.graph_seed
        np.testing.assert_array_equal(left.graph, right.graph)
        np.testing.assert_array_equal(left.initial_state.attention, right.initial_state.attention)


@pytest.mark.parametrize(
    "bad_specifications",
    [
        specifications()[:2],
        specifications()
        + (TopologySpecification(topology_label="EXTRA", kind="random", k=2),),
    ],
)
def test_prepare_rejects_label_mismatch(bad_specifications):
    with pytest.raises(ValueError, match="exactly match"):
        prepare_paired_treatments(
            plan=plan(),
            specifications=bad_specifications,
            initial_conditions=initial_conditions(),
            parameters=parameters(),
        )


def test_prepare_rejects_inconsistent_k():
    bad = (
        TopologySpecification(topology_label="R", kind="random", k=2),
        TopologySpecification(topology_label="SW", kind="small_world", k=4, p_sw=0.2),
        TopologySpecification(topology_label="SF", kind="hub_dominated", k=2, a0=1.0),
    )
    with pytest.raises(ValueError, match="same k"):
        prepare_paired_treatments(
            plan=plan(),
            specifications=bad,
            initial_conditions=initial_conditions(),
            parameters=parameters(),
        )


def test_prepare_rejects_shock_dimension_mismatch():
    with pytest.raises(ValueError, match="shock dimension"):
        prepare_paired_treatments(
            plan=plan(n_agents=7),
            specifications=specifications(),
            initial_conditions=initial_conditions(n_agents=8),
            parameters=parameters(),
        )


def test_prepare_rejects_initial_positions_outside_inventory_bound():
    with pytest.raises(ValueError, match="inventory bound"):
        prepare_paired_treatments(
            plan=plan(),
            specifications=specifications(),
            initial_conditions=initial_conditions(position=3.0),
            parameters=parameters(),
        )


@pytest.mark.parametrize("bad_field", ["plan", "initial_conditions", "parameters"])
def test_prepare_rejects_wrong_top_level_types(bad_field):
    kwargs = {
        "plan": plan(),
        "specifications": specifications(),
        "initial_conditions": initial_conditions(),
        "parameters": parameters(),
    }
    kwargs[bad_field] = "invalid"

    with pytest.raises(TypeError):
        prepare_paired_treatments(**kwargs)
