"""Tests for structural-only matched benchmark ensemble validation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.experiments.refined import (
    TopologySpecification,
    derive_graph_seed,
    prepare_paired_replication,
    run_structural_ensemble,
)
from src.model.refined import RefinedParameters


def specifications() -> tuple[TopologySpecification, ...]:
    return (
        TopologySpecification(topology_label="R", kind="random", k=2),
        TopologySpecification(topology_label="SW", kind="small_world", k=2, p_sw=0.25),
        TopologySpecification(topology_label="SF", kind="hub_dominated", k=2, a0=1.0),
    )


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


def result(*, experiment_seed: int = 1234, n_replications: int = 5):
    return run_structural_ensemble(
        experiment_seed=experiment_seed,
        n_replications=n_replications,
        n_agents=8,
        q=2,
        specifications=specifications(),
    )


def test_graph_seed_helper_matches_paired_market_plan() -> None:
    paired = prepare_paired_replication(
        experiment_seed=1234,
        replication_id=3,
        topology_labels=("R", "SW", "SF"),
        n_periods=1,
        n_agents=8,
        parameters=parameters(),
    )

    for label in paired.topology_labels:
        assert derive_graph_seed(
            experiment_seed=1234,
            replication_id=3,
            topology_label=label,
        ) == paired.graph_seed_for(label)


def test_graph_seed_is_topology_specific() -> None:
    seeds = {
        derive_graph_seed(
            experiment_seed=1234,
            replication_id=7,
            topology_label=label,
        )
        for label in ("R", "SW", "SF")
    }
    assert len(seeds) == 3


def test_structural_runner_returns_balanced_raw_records() -> None:
    ensemble = result(n_replications=6)
    assert ensemble.topology_labels == ("R", "SW", "SF")
    assert ensemble.n_replications == 6
    assert len(ensemble.records) == 18
    assert all(len(ensemble.records_for(label)) == 6 for label in ensemble.topology_labels)


def test_structural_records_preserve_matched_link_budget() -> None:
    ensemble = result()
    for record in ensemble.records:
        diagnostics = record.diagnostics
        assert diagnostics.n_agents == 8
        assert diagnostics.total_links == 16
        assert diagnostics.mean_out_degree == 2.0
        assert diagnostics.hub_q == 2


def test_structural_runner_is_reproducible() -> None:
    first = result()
    second = result()
    assert first.records == second.records
    assert first.specifications == second.specifications


def test_different_experiment_seed_changes_graph_seeds() -> None:
    first = result(experiment_seed=111)
    second = result(experiment_seed=222)
    first_seeds = tuple(record.graph_seed for record in first.records)
    second_seeds = tuple(record.graph_seed for record in second.records)
    assert first_seeds != second_seeds


def test_specification_order_does_not_change_label_replication_records() -> None:
    ordered = specifications()
    reversed_specs = tuple(reversed(ordered))
    first = run_structural_ensemble(
        experiment_seed=555,
        n_replications=4,
        n_agents=8,
        q=2,
        specifications=ordered,
    )
    second = run_structural_ensemble(
        experiment_seed=555,
        n_replications=4,
        n_agents=8,
        q=2,
        specifications=reversed_specs,
    )

    for label in ("R", "SW", "SF"):
        left = first.records_for(label)
        right = second.records_for(label)
        assert left == right


def test_metric_values_preserve_replication_order() -> None:
    ensemble = result(n_replications=4)
    values = ensemble.metric_values("R", "in_degree_gini")
    expected = np.array(
        [record.diagnostics.in_degree_gini for record in ensemble.records_for("R")]
    )
    np.testing.assert_array_equal(values, expected)
    assert values.shape == (4,)


def test_summary_matches_raw_numpy_statistics() -> None:
    ensemble = result(n_replications=7)
    values = ensemble.metric_values("SF", "hub_link_share")
    summary = ensemble.summary_for("SF").hub_link_share
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])

    assert summary.count == 7
    assert summary.mean == pytest.approx(float(np.mean(values)))
    assert summary.std == pytest.approx(float(np.std(values, ddof=0)))
    assert summary.minimum == pytest.approx(float(np.min(values)))
    assert summary.q25 == pytest.approx(float(q25))
    assert summary.median == pytest.approx(float(median))
    assert summary.q75 == pytest.approx(float(q75))
    assert summary.maximum == pytest.approx(float(np.max(values)))


def test_summary_does_not_replace_raw_records() -> None:
    ensemble = result(n_replications=3)
    before = ensemble.records
    _ = ensemble.summary_for("SW")
    assert ensemble.records == before
    assert len(ensemble.records_for("SW")) == 3


def test_unknown_topology_label_rejected() -> None:
    with pytest.raises(KeyError, match="unknown topology label"):
        result().records_for("UNKNOWN")


def test_unknown_metric_rejected() -> None:
    with pytest.raises(KeyError, match="unknown structural metric"):
        result().metric_values("R", "not_a_metric")


def test_runner_public_signature_contains_no_market_randomness() -> None:
    names = set(inspect.signature(run_structural_ensemble).parameters)
    assert names == {
        "experiment_seed",
        "n_replications",
        "n_agents",
        "q",
        "specifications",
    }
    assert "parameters" not in names
    assert "shock_seed" not in names
    assert "n_periods" not in names
    assert "initial_state_seed" not in names


@pytest.mark.parametrize("n_replications", [0, -1, 1.5, True])
def test_runner_rejects_invalid_replication_count(n_replications) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_structural_ensemble(
            experiment_seed=1,
            n_replications=n_replications,
            n_agents=8,
            q=2,
            specifications=specifications(),
        )


@pytest.mark.parametrize("q", [0, 9, 1.5, True])
def test_runner_rejects_invalid_hub_count(q) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_structural_ensemble(
            experiment_seed=1,
            n_replications=2,
            n_agents=8,
            q=q,
            specifications=specifications(),
        )


@pytest.mark.parametrize("experiment_seed", [-1, 1.5, True])
def test_runner_rejects_invalid_experiment_seed(experiment_seed) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_structural_ensemble(
            experiment_seed=experiment_seed,
            n_replications=2,
            n_agents=8,
            q=2,
            specifications=specifications(),
        )


def test_runner_rejects_mismatched_k() -> None:
    bad = (
        TopologySpecification(topology_label="R", kind="random", k=2),
        TopologySpecification(topology_label="SW", kind="small_world", k=4, p_sw=0.2),
    )
    with pytest.raises(ValueError, match="same k"):
        run_structural_ensemble(
            experiment_seed=1,
            n_replications=2,
            n_agents=8,
            q=2,
            specifications=bad,
        )


def test_runner_rejects_duplicate_labels() -> None:
    bad = (
        TopologySpecification(topology_label="R", kind="random", k=2),
        TopologySpecification(topology_label="R", kind="small_world", k=2, p_sw=0.2),
    )
    with pytest.raises(ValueError, match="labels must be unique"):
        run_structural_ensemble(
            experiment_seed=1,
            n_replications=2,
            n_agents=8,
            q=2,
            specifications=bad,
        )


def test_runner_rejects_single_specification() -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_structural_ensemble(
            experiment_seed=1,
            n_replications=2,
            n_agents=8,
            q=2,
            specifications=specifications()[:1],
        )


def test_runner_rejects_non_specification_entry() -> None:
    with pytest.raises(TypeError, match="TopologySpecification"):
        run_structural_ensemble(
            experiment_seed=1,
            n_replications=2,
            n_agents=8,
            q=2,
            specifications=(specifications()[0], "invalid"),
        )
