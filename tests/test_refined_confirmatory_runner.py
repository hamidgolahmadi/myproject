from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.confirmatory_runner import (
    run_paired_confirmatory_replication,
    run_paired_confirmatory_smoke,
    write_paired_confirmatory_smoke,
)
from src.experiments.refined.market_calibration import (
    MarketEvaluationCalibration,
    MarketEvaluationCalibrationProtocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _small_baseline():
    base = first_refined_baseline_specification()
    return replace(base, n_agents=8, k=2, horizon=8, hub_q=2, p_sw=0.25)


def _small_calibration():
    protocol = MarketEvaluationCalibrationProtocol(
        scale_calibration_seed=99101,
        threshold_calibration_seed=99102,
        n_scale_replications=2,
        n_threshold_replications=2,
        horizon=8,
        burn_in=0,
        rolling_window=3,
        calibration_alpha=0.0,
        cid_weights=CIDWeights.equal(),
        cid_peak_quantile=0.95,
        stabilisation_length=2,
    )
    return MarketEvaluationCalibration(
        protocol=protocol,
        reference_scales=CIDReferenceScales(
            return_scale=0.01,
            belief_scale=0.01,
            order_flow_scale=0.2,
        ),
        cid_weights=protocol.cid_weights,
        cid_threshold=2.0,
    )


@pytest.fixture(scope="module")
def baseline_records():
    return run_paired_confirmatory_replication(
        experiment_seed=88001,
        replication_id=0,
        baseline=_small_baseline(),
        calibration=_small_calibration(),
    )


@pytest.fixture(scope="module")
def alpha0_records():
    return run_paired_confirmatory_replication(
        experiment_seed=88001,
        replication_id=0,
        regime="alpha0_control",
        alpha_override=0.0,
        baseline=_small_baseline(),
        calibration=_small_calibration(),
    )


def test_replication_returns_exactly_three_topologies(baseline_records):
    assert tuple(record.topology_label for record in baseline_records) == ("R", "SW", "SF")


def test_common_random_number_seeds_are_shared_within_replication(baseline_records):
    assert len({record.shock_seed for record in baseline_records}) == 1
    assert len({record.initial_state_seed for record in baseline_records}) == 1


def test_graph_seeds_are_topology_specific(baseline_records):
    assert len({record.graph_seed for record in baseline_records}) == 3


def test_baseline_alpha_is_preserved(baseline_records):
    assert all(record.alpha == pytest.approx(0.75) for record in baseline_records)


def test_alpha0_override_is_applied_to_all_treatments(alpha0_records):
    assert {record.alpha for record in alpha0_records} == {0.0}


def test_alpha0_economic_paths_are_exactly_topology_null(alpha0_records):
    assert len({record.economic_path_fingerprint for record in alpha0_records}) == 1


def test_alpha0_market_metrics_are_exactly_equal(alpha0_records):
    fields = (
        "return_volatility",
        "rms_mispricing",
        "maximum_absolute_mispricing",
        "mean_absolute_order_flow_per_agent",
        "mean_absolute_return",
        "time_averaged_belief_variance",
        "peak_cid",
        "cid_exceedance_duration_share",
    )
    for field in fields:
        values = [getattr(record, field) for record in alpha0_records]
        assert values[1:] == values[:-1]


def test_alpha0_structural_diagnostics_remain_valid(alpha0_records):
    for record in alpha0_records:
        assert np.isfinite(record.in_degree_gini)
        assert np.isfinite(record.hub_link_share)
        assert np.isfinite(record.global_clustering)
        assert 0.0 <= record.largest_component_share <= 1.0


def test_records_contain_finite_market_and_mechanism_values(baseline_records):
    nonnegative_fields = (
        "return_volatility",
        "rms_mispricing",
        "maximum_absolute_mispricing",
        "mean_absolute_order_flow_per_agent",
        "mean_absolute_return",
        "time_averaged_belief_variance",
        "peak_cid",
        "cid_exceedance_duration_share",
        "in_degree_gini",
        "hub_link_share",
        "global_clustering",
        "average_path_length_lcc",
        "largest_component_share",
        "mean_attention_entropy",
        "mean_effective_sources",
        "mean_influence_hhi",
        "mean_hub_influence_share",
        "mean_attention_overlap",
        "mean_attention_mobility",
    )
    for record in baseline_records:
        for field in nonnegative_fields:
            value = getattr(record, field)
            assert np.isfinite(value)
            assert value >= 0.0


def test_probability_like_record_fields_lie_in_unit_interval(baseline_records):
    fields = (
        "cid_exceedance_duration_share",
        "in_degree_gini",
        "hub_link_share",
        "global_clustering",
        "largest_component_share",
        "mean_attention_entropy",
        "mean_influence_hhi",
        "mean_hub_influence_share",
        "mean_attention_overlap",
    )
    for record in baseline_records:
        for field in fields:
            assert 0.0 <= getattr(record, field) <= 1.0


def test_stabilisation_flags_are_complementary(baseline_records):
    for record in baseline_records:
        assert record.stabilised != record.right_censored
        if record.stabilised:
            assert record.stabilisation_period is not None
        else:
            assert record.stabilisation_period is None


def test_invalid_alpha_override_is_rejected():
    with pytest.raises(ValueError):
        run_paired_confirmatory_replication(
            experiment_seed=1,
            replication_id=0,
            alpha_override=1.1,
            baseline=_small_baseline(),
            calibration=_small_calibration(),
        )


def test_boolean_alpha_override_is_rejected():
    with pytest.raises(TypeError):
        run_paired_confirmatory_replication(
            experiment_seed=1,
            replication_id=0,
            alpha_override=True,
            baseline=_small_baseline(),
            calibration=_small_calibration(),
        )


def test_horizon_mismatch_is_rejected():
    calibration = _small_calibration()
    bad_baseline = replace(_small_baseline(), horizon=9)
    with pytest.raises(ValueError, match="horizon"):
        run_paired_confirmatory_replication(
            experiment_seed=1,
            replication_id=0,
            baseline=bad_baseline,
            calibration=calibration,
        )


def test_smoke_has_two_regimes_and_six_records_per_replication():
    result = run_paired_confirmatory_smoke(
        experiment_seed=88002,
        n_replications=1,
        baseline=_small_baseline(),
        calibration=_small_calibration(),
    )
    assert len(result.records) == 6
    assert {record.regime for record in result.records} == {"baseline", "alpha0_control"}


def test_smoke_rejects_invalid_replication_count():
    with pytest.raises(ValueError):
        run_paired_confirmatory_smoke(
            n_replications=0,
            baseline=_small_baseline(),
            calibration=_small_calibration(),
        )


def test_smoke_writer_persists_csv_and_metadata(tmp_path):
    result = run_paired_confirmatory_smoke(
        experiment_seed=88003,
        n_replications=1,
        baseline=_small_baseline(),
        calibration=_small_calibration(),
    )
    records_path, metadata_path = write_paired_confirmatory_smoke(result, outdir=tmp_path)
    assert records_path.exists()
    assert metadata_path.exists()
    assert len(records_path.read_text().strip().splitlines()) == 7


def test_smoke_metadata_is_explicitly_not_final(tmp_path):
    result = run_paired_confirmatory_smoke(
        experiment_seed=88004,
        n_replications=1,
        baseline=_small_baseline(),
        calibration=_small_calibration(),
    )
    _, metadata_path = write_paired_confirmatory_smoke(result, outdir=tmp_path)
    payload = json.loads(metadata_path.read_text())
    assert payload["final_confirmatory"] is False
    assert payload["interpretation_guard"] == "do not rank topologies from this smoke"
    assert payload["calibration"]["c_CID"] == 2.0


def test_smoke_metadata_records_actual_small_calibration(tmp_path):
    result = run_paired_confirmatory_smoke(
        experiment_seed=88005,
        n_replications=1,
        baseline=_small_baseline(),
        calibration=_small_calibration(),
    )
    _, metadata_path = write_paired_confirmatory_smoke(result, outdir=tmp_path)
    payload = json.loads(metadata_path.read_text())
    assert payload["calibration"]["rolling_window"] == 3
    assert payload["baseline_alpha"] == pytest.approx(0.75)


def test_confirmatory_cli_help_runs_without_pythonpath():
    completed = subprocess.run(
        [sys.executable, "scripts/run_refined_confirmatory_smoke.py", "--help"],
        cwd=REPO_ROOT,
        env={key: value for key, value in __import__("os").environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--replications" in completed.stdout
