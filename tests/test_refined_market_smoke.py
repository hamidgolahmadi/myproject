from __future__ import annotations

import numpy as np
import pytest

from src.experiments.refined.baseline_specification import (
    RefinedBaselineCandidate,
    first_refined_baseline_candidate,
    generate_neutral_nonnetwork_initial_conditions,
)
from src.experiments.refined.market_smoke import (
    MarketScaleSmokeProtocol,
    MarketScaleSmokeRecord,
    diagnose_market_scale,
    run_first_refined_baseline_scale_smoke,
    summarise_market_scale_smoke,
)
from src.model.refined import (
    SimulationResult,
    generate_shock_path,
    simulate_shock_path,
    uniform_attention_from_graph,
)
from src.model.refined.state import initialise_state


def _valid_record(**overrides) -> MarketScaleSmokeRecord:
    values = dict(
        replication_id=0,
        topology_label="R",
        return_std=0.01,
        mean_abs_return=0.008,
        max_abs_return=0.03,
        rms_mispricing=0.02,
        max_abs_mispricing=0.05,
        mean_abs_flow_per_agent=0.1,
        rms_flow_per_agent=0.12,
        desired_action_abs_p95=0.4,
        desired_action_saturation_fraction=0.01,
        execution_projection_fraction=0.02,
        inventory_boundary_fraction=0.03,
        median_local_reputation_std=0.001,
        max_local_reputation_std=0.004,
        median_reputation_scale_to_sigma0=1000.0,
        mean_attention_mobility=0.02,
        max_attention_mobility=0.08,
        final_attention_distance_from_initial=0.1,
    )
    values.update(overrides)
    return MarketScaleSmokeRecord(**values)


def _small_simulation(n_periods: int = 4) -> tuple[SimulationResult, np.ndarray, RefinedBaselineCandidate]:
    candidate = RefinedBaselineCandidate(n_agents=4, k=2, horizon=n_periods, hub_q=1)
    parameters = candidate.parameters
    graph = np.asarray(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    initial = generate_neutral_nonnetwork_initial_conditions(
        n_agents=4,
        parameters=parameters,
        initial_state_seed=123,
    )
    state = initialise_state(
        theta=initial.theta,
        beliefs=initial.beliefs,
        positions=initial.positions,
        price=initial.price,
        reputation=initial.reputation,
        attention=uniform_attention_from_graph(graph),
        graph=graph,
        x_bar=parameters.x_bar,
    )
    shocks = generate_shock_path(
        n_periods=n_periods,
        n_agents=4,
        parameters=parameters,
        shock_seed=456,
    )
    result = simulate_shock_path(
        state,
        shocks,
        graph,
        parameters,
        adaptive_attention=True,
    )
    return result, graph, candidate


def test_market_scale_smoke_protocol_defaults() -> None:
    protocol = MarketScaleSmokeProtocol()
    assert protocol.experiment_seed == 2026090203
    assert protocol.n_replications == 5


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_market_scale_smoke_protocol_rejects_bad_seed(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        MarketScaleSmokeProtocol(experiment_seed=value)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_market_scale_smoke_protocol_rejects_bad_replications(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        MarketScaleSmokeProtocol(n_replications=value)


def test_market_scale_smoke_record_accepts_valid_values() -> None:
    record = _valid_record()
    assert record.topology_label == "R"
    assert record.return_std == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("desired_action_saturation_fraction", 1.1),
        ("execution_projection_fraction", 1.1),
        ("inventory_boundary_fraction", 1.1),
    ],
)
def test_market_scale_smoke_record_rejects_fraction_above_one(field, value) -> None:
    with pytest.raises(ValueError):
        _valid_record(**{field: value})


@pytest.mark.parametrize(
    "field",
    ["return_std", "rms_mispricing", "mean_attention_mobility"],
)
def test_market_scale_smoke_record_rejects_negative_metric(field) -> None:
    with pytest.raises(ValueError):
        _valid_record(**{field: -0.1})


def test_summarise_market_scale_smoke_exact_two_record_summary() -> None:
    first = _valid_record(return_std=1.0)
    second = _valid_record(replication_id=1, topology_label="SW", return_std=3.0)
    summaries = summarise_market_scale_smoke((first, second))
    return_summary = next(item for item in summaries if item.metric == "return_std")
    assert return_summary.count == 2
    assert return_summary.mean == pytest.approx(2.0)
    assert return_summary.median == pytest.approx(2.0)
    assert return_summary.minimum == pytest.approx(1.0)
    assert return_summary.maximum == pytest.approx(3.0)


def test_summarise_market_scale_smoke_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarise_market_scale_smoke(())


def test_summarise_market_scale_smoke_rejects_wrong_record_type() -> None:
    with pytest.raises(TypeError):
        summarise_market_scale_smoke((_valid_record(), object()))


def test_diagnose_market_scale_rejects_wrong_result_type() -> None:
    with pytest.raises(TypeError):
        diagnose_market_scale(
            object(),
            graph=np.ones((2, 2)),
            x_bar=1.0,
            sigma_0=1e-6,
            replication_id=0,
            topology_label="R",
        )


def test_diagnose_market_scale_requires_two_periods() -> None:
    result, graph, candidate = _small_simulation(n_periods=2)
    short = SimulationResult(states=result.states[:2], period_outputs=result.period_outputs[:1])
    with pytest.raises(ValueError, match="at least two"):
        diagnose_market_scale(
            short,
            graph=graph,
            x_bar=candidate.parameters.x_bar,
            sigma_0=candidate.parameters.sigma_0,
            replication_id=0,
            topology_label="R",
        )


@pytest.mark.parametrize(
    ("x_bar", "sigma_0"),
    [(0.0, 1e-6), (-1.0, 1e-6), (5.0, 0.0), (5.0, -1.0)],
)
def test_diagnose_market_scale_rejects_invalid_positive_scales(x_bar, sigma_0) -> None:
    result, graph, _ = _small_simulation()
    with pytest.raises(ValueError):
        diagnose_market_scale(
            result,
            graph=graph,
            x_bar=x_bar,
            sigma_0=sigma_0,
            replication_id=0,
            topology_label="R",
        )


def test_diagnose_market_scale_on_real_short_path_is_finite_and_bounded() -> None:
    result, graph, candidate = _small_simulation()
    record = diagnose_market_scale(
        result,
        graph=graph,
        x_bar=candidate.parameters.x_bar,
        sigma_0=candidate.parameters.sigma_0,
        replication_id=7,
        topology_label="R",
    )
    assert record.return_std > 0.0
    assert record.rms_flow_per_agent > 0.0
    assert 0.0 <= record.desired_action_saturation_fraction <= 1.0
    assert 0.0 <= record.execution_projection_fraction <= 1.0
    assert 0.0 <= record.inventory_boundary_fraction <= 1.0
    assert record.max_attention_mobility >= record.mean_attention_mobility


def test_run_first_refined_baseline_scale_smoke_tiny_shape_and_labels() -> None:
    candidate = RefinedBaselineCandidate(n_agents=8, k=2, horizon=8, hub_q=2)
    protocol = MarketScaleSmokeProtocol(experiment_seed=99, n_replications=2)
    result = run_first_refined_baseline_scale_smoke(
        protocol=protocol,
        candidate=candidate,
    )
    assert len(result.records) == 6
    assert {record.topology_label for record in result.records} == {"R", "SW", "SF"}
    assert len(result.pooled_summary) == 17
    assert all(summary.count == 6 for summary in result.pooled_summary)


def test_run_first_refined_baseline_scale_smoke_is_reproducible() -> None:
    candidate = RefinedBaselineCandidate(n_agents=8, k=2, horizon=8, hub_q=2)
    protocol = MarketScaleSmokeProtocol(experiment_seed=101, n_replications=1)
    first = run_first_refined_baseline_scale_smoke(protocol=protocol, candidate=candidate)
    second = run_first_refined_baseline_scale_smoke(protocol=protocol, candidate=candidate)
    assert first.records == second.records
    assert first.pooled_summary == second.pooled_summary


def test_run_first_refined_baseline_scale_smoke_rejects_wrong_protocol_type() -> None:
    with pytest.raises(TypeError):
        run_first_refined_baseline_scale_smoke(
            protocol=object(),
            candidate=first_refined_baseline_candidate(),
        )


def test_run_first_refined_baseline_scale_smoke_rejects_wrong_candidate_type() -> None:
    with pytest.raises(TypeError):
        run_first_refined_baseline_scale_smoke(
            protocol=MarketScaleSmokeProtocol(n_replications=1),
            candidate=object(),
        )
