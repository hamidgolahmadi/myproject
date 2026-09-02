from dataclasses import replace

import numpy as np
import pytest

from src.experiments.refined.baseline_specification import first_refined_baseline_candidate
from src.experiments.refined.market_smoke import SmokeMetricSummary
from src.experiments.refined.sigma0_sensitivity import (
    Sigma0SensitivityProtocol,
    Sigma0SensitivityResult,
    Sigma0SensitivityRow,
    _selected_rows,
    run_sigma0_sensitivity_smoke,
)


def _small_candidate():
    base = first_refined_baseline_candidate()
    return replace(
        base,
        n_agents=8,
        k=2,
        horizon=8,
        hub_q=2,
        p_sw=0.25,
    )


def _summary(metric: str, value: float = 1.0) -> SmokeMetricSummary:
    return SmokeMetricSummary(
        metric=metric,
        count=3,
        mean=value,
        median=value,
        minimum=value,
        maximum=value,
    )


def test_default_protocol_values():
    protocol = Sigma0SensitivityProtocol()
    assert protocol.experiment_seed == 2026090203
    assert protocol.n_replications == 5
    assert protocol.sigma0_values == (1e-6, 1e-4, 5e-4, 1e-3, 2e-3)


def test_protocol_preserves_sigma0_order():
    protocol = Sigma0SensitivityProtocol(sigma0_values=(0.002, 0.0005, 0.001))
    assert protocol.sigma0_values == (0.002, 0.0005, 0.001)


def test_protocol_rejects_single_sigma0_value():
    with pytest.raises(ValueError):
        Sigma0SensitivityProtocol(sigma0_values=(1e-3,))


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
def test_protocol_rejects_invalid_sigma0_values(value):
    with pytest.raises((TypeError, ValueError)):
        Sigma0SensitivityProtocol(sigma0_values=(1e-6, value))


def test_protocol_rejects_duplicate_sigma0_values():
    with pytest.raises(ValueError):
        Sigma0SensitivityProtocol(sigma0_values=(1e-4, 1e-4))


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_protocol_rejects_invalid_replication_count(value):
    with pytest.raises((TypeError, ValueError)):
        Sigma0SensitivityProtocol(n_replications=value)


def test_sensitivity_row_accepts_known_metric():
    row = Sigma0SensitivityRow(
        sigma_0=1e-3,
        metric="return_std",
        count=3,
        mean=1.0,
        median=1.0,
        minimum=0.5,
        maximum=1.5,
    )
    assert row.sigma_0 == 1e-3
    assert row.count == 3


def test_sensitivity_row_rejects_unknown_metric():
    with pytest.raises(ValueError):
        Sigma0SensitivityRow(
            sigma_0=1e-3,
            metric="not_a_metric",
            count=3,
            mean=1.0,
            median=1.0,
            minimum=0.5,
            maximum=1.5,
        )


def test_selected_rows_contains_only_reported_subset():
    metrics = (
        "return_std",
        "rms_mispricing",
        "rms_flow_per_agent",
        "desired_action_abs_p95",
        "execution_projection_fraction",
        "inventory_boundary_fraction",
        "median_local_reputation_std",
        "median_reputation_scale_to_sigma0",
        "mean_attention_mobility",
        "max_attention_mobility",
        "final_attention_distance_from_initial",
    )
    rows = _selected_rows(1e-3, tuple(_summary(metric) for metric in metrics))
    assert len(rows) == 11
    assert {row.metric for row in rows} == set(metrics)


def _small_result() -> Sigma0SensitivityResult:
    return run_sigma0_sensitivity_smoke(
        protocol=Sigma0SensitivityProtocol(
            experiment_seed=1234,
            n_replications=1,
            sigma0_values=(1e-4, 1e-3),
        ),
        candidate=_small_candidate(),
    )


def test_result_rows_for_known_sigma0():
    result = _small_result()
    rows = result.rows_for(1e-4)
    assert len(rows) == 11
    assert all(row.sigma_0 == 1e-4 for row in rows)


def test_result_rows_for_unknown_sigma0_rejected():
    result = _small_result()
    with pytest.raises(KeyError):
        result.rows_for(0.123)


def test_result_metric_values_follow_sigma0_order():
    result = _small_result()
    values = result.metric_values("mean_attention_mobility")
    assert tuple(value[0] for value in values) == (1e-4, 1e-3)


def test_result_metric_values_unknown_metric_rejected():
    result = _small_result()
    with pytest.raises(KeyError):
        result.metric_values("unknown")


def test_small_integration_smoke_returns_one_result_per_sigma0():
    result = _small_result()
    assert len(result.smoke_results) == 2
    assert len(result.pooled_rows) == 22


def test_small_integration_uses_requested_sigma0_in_each_smoke():
    result = _small_result()
    realised = tuple(smoke.candidate.parameters.sigma_0 for smoke in result.smoke_results)
    assert realised == (1e-4, 1e-3)


def test_small_integration_does_not_mutate_base_candidate():
    candidate = _small_candidate()
    original_sigma0 = candidate.parameters.sigma_0
    run_sigma0_sensitivity_smoke(
        protocol=Sigma0SensitivityProtocol(
            experiment_seed=5678,
            n_replications=1,
            sigma0_values=(1e-4, 1e-3),
        ),
        candidate=candidate,
    )
    assert candidate.parameters.sigma_0 == original_sigma0


def test_small_integration_preserves_common_record_identifiers_across_sigma0():
    result = _small_result()
    identifiers = []
    for smoke in result.smoke_results:
        identifiers.append(
            tuple((record.replication_id, record.topology_label) for record in smoke.records)
        )
    assert identifiers[0] == identifiers[1]
