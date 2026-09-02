from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.market_calibration import (
    MarketEvaluationCalibrationProtocol,
    estimate_cid_threshold,
    estimate_reference_scales,
)
from src.experiments.refined.market_calibration_run import (
    calibration_configuration_fingerprint,
    load_reference_scales,
    run_final_market_calibration,
    run_scale_calibration_stage,
    run_threshold_calibration_stage,
)
from src.experiments.refined.no_social_calibration_paths import (
    no_social_component_path,
    no_social_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _small_baseline():
    base = first_refined_baseline_specification()
    return replace(base, n_agents=8, k=2, horizon=8, hub_q=2, p_sw=0.25)


def _small_protocol():
    return MarketEvaluationCalibrationProtocol(
        scale_calibration_seed=9101,
        threshold_calibration_seed=9102,
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


def _direct_paths(seed: int, count: int, *, adaptive_attention: bool):
    baseline = _small_baseline()
    protocol = _small_protocol()
    return tuple(
        no_social_component_path(
            experiment_seed=seed,
            replication_id=replication_id,
            baseline=baseline,
            protocol=protocol,
            adaptive_attention=adaptive_attention,
        )
        for replication_id in range(count)
    )


def test_no_social_parameters_change_only_alpha():
    baseline = _small_baseline()
    no_social = no_social_parameters(baseline)
    assert no_social.alpha == 0.0
    assert replace(no_social, alpha=baseline.parameters.alpha) == baseline.parameters


def test_no_social_component_path_rejects_non_bool_attention_flag():
    with pytest.raises(TypeError):
        no_social_component_path(
            experiment_seed=1,
            replication_id=0,
            baseline=_small_baseline(),
            protocol=_small_protocol(),
            adaptive_attention=1,
        )


def test_alpha0_adaptive_and_fixed_attention_have_exactly_equal_cid_components():
    baseline = _small_baseline()
    protocol = _small_protocol()
    adaptive = no_social_component_path(
        experiment_seed=777,
        replication_id=0,
        baseline=baseline,
        protocol=protocol,
        adaptive_attention=True,
    )
    fixed = no_social_component_path(
        experiment_seed=777,
        replication_id=0,
        baseline=baseline,
        protocol=protocol,
        adaptive_attention=False,
    )
    assert adaptive == fixed


def test_configuration_fingerprint_is_deterministic():
    protocol = _small_protocol()
    baseline = _small_baseline()
    assert calibration_configuration_fingerprint(protocol, baseline) == calibration_configuration_fingerprint(
        protocol, baseline
    )


def test_configuration_fingerprint_changes_when_baseline_changes():
    protocol = _small_protocol()
    baseline = _small_baseline()
    changed = replace(
        baseline,
        parameters=replace(baseline.parameters, sigma_0=baseline.parameters.sigma_0 * 2),
    )
    assert calibration_configuration_fingerprint(protocol, baseline) != calibration_configuration_fingerprint(
        protocol, changed
    )


def test_scale_stage_matches_existing_reference_scale_estimator(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    direct = _direct_paths(protocol.scale_calibration_seed, protocol.n_scale_replications, adaptive_attention=False)
    expected = estimate_reference_scales(direct, protocol=protocol)
    actual = run_scale_calibration_stage(
        output_dir=tmp_path,
        protocol=protocol,
        baseline=baseline,
    )
    assert actual == expected


def test_scale_stage_writes_one_checkpoint_per_replication(tmp_path):
    protocol = _small_protocol()
    run_scale_calibration_stage(output_dir=tmp_path, protocol=protocol, baseline=_small_baseline())
    checkpoints = sorted((tmp_path / "scale").glob("replication_*.npz"))
    assert len(checkpoints) == protocol.n_scale_replications


def test_scale_artifact_is_complete_but_not_final_calibration(tmp_path):
    run_scale_calibration_stage(output_dir=tmp_path, protocol=_small_protocol(), baseline=_small_baseline())
    payload = json.loads((tmp_path / "reference_scales.json").read_text())
    assert payload["stage_complete"] is True
    assert payload["final_calibration"] is False
    assert payload["reference_scales"]["c_ret"] > 0.0
    assert payload["reference_scales"]["c_bel"] > 0.0
    assert payload["reference_scales"]["c_F"] > 0.0


def test_load_reference_scales_round_trip(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    expected = run_scale_calibration_stage(output_dir=tmp_path, protocol=protocol, baseline=baseline)
    actual = load_reference_scales(tmp_path, protocol=protocol, baseline=baseline)
    assert actual == expected


def test_scale_resume_reuses_all_existing_checkpoints(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    run_scale_calibration_stage(output_dir=tmp_path, protocol=protocol, baseline=baseline)
    events = []
    run_scale_calibration_stage(
        output_dir=tmp_path,
        protocol=protocol,
        baseline=baseline,
        resume=True,
        progress=lambda stage, completed, total, reused: events.append((stage, completed, total, reused)),
    )
    assert len(events) == protocol.n_scale_replications
    assert all(event[3] is True for event in events)


def test_scale_resume_rejects_stale_baseline_checkpoint(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    run_scale_calibration_stage(output_dir=tmp_path, protocol=protocol, baseline=baseline)
    changed = replace(
        baseline,
        parameters=replace(baseline.parameters, sigma_0=baseline.parameters.sigma_0 * 2),
    )
    with pytest.raises(ValueError, match="specification mismatch"):
        run_scale_calibration_stage(
            output_dir=tmp_path,
            protocol=protocol,
            baseline=changed,
            resume=True,
        )


def test_threshold_stage_matches_existing_peak_quantile_estimator(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    scales = run_scale_calibration_stage(output_dir=tmp_path, protocol=protocol, baseline=baseline)
    threshold_paths = _direct_paths(
        protocol.threshold_calibration_seed,
        protocol.n_threshold_replications,
        adaptive_attention=False,
    )
    expected = estimate_cid_threshold(threshold_paths, protocol=protocol, scales=scales)
    result = run_threshold_calibration_stage(
        output_dir=tmp_path,
        scales=scales,
        protocol=protocol,
        baseline=baseline,
    )
    assert result.calibration.cid_threshold == pytest.approx(expected, rel=0.0, abs=0.0)


def test_threshold_stage_writes_one_checkpoint_per_replication(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    scales = run_scale_calibration_stage(output_dir=tmp_path, protocol=protocol, baseline=baseline)
    run_threshold_calibration_stage(
        output_dir=tmp_path,
        scales=scales,
        protocol=protocol,
        baseline=baseline,
    )
    checkpoints = sorted((tmp_path / "threshold").glob("replication_*.json"))
    assert len(checkpoints) == protocol.n_threshold_replications


def test_threshold_resume_rejects_different_reference_scales(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    scales = run_scale_calibration_stage(output_dir=tmp_path, protocol=protocol, baseline=baseline)
    run_threshold_calibration_stage(
        output_dir=tmp_path,
        scales=scales,
        protocol=protocol,
        baseline=baseline,
    )
    changed_scales = CIDReferenceScales(
        return_scale=scales.return_scale * 1.01,
        belief_scale=scales.belief_scale,
        order_flow_scale=scales.order_flow_scale,
    )
    with pytest.raises(ValueError, match="reference-scale mismatch"):
        run_threshold_calibration_stage(
            output_dir=tmp_path,
            scales=changed_scales,
            protocol=protocol,
            baseline=baseline,
            resume=True,
        )


def test_final_runner_writes_final_calibration_artifact(tmp_path):
    result = run_final_market_calibration(
        output_dir=tmp_path,
        protocol=_small_protocol(),
        baseline=_small_baseline(),
    )
    artifact = tmp_path / "market_evaluation_calibration.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["final_calibration"] is True
    assert payload["calibration_alpha"] == 0.0
    assert payload["adaptive_attention_during_calibration"] is False
    assert payload["reference_scales"]["c_ret"] == pytest.approx(
        result.calibration.reference_scales.return_scale
    )
    assert payload["c_CID"] == pytest.approx(result.calibration.cid_threshold)


def test_final_artifact_records_inactive_guardrails(tmp_path):
    run_final_market_calibration(
        output_dir=tmp_path,
        protocol=_small_protocol(),
        baseline=_small_baseline(),
    )
    payload = json.loads((tmp_path / "market_evaluation_calibration.json").read_text())
    assert payload["component_guardrails"] == {
        "return": None,
        "belief": None,
        "order_flow": None,
    }


def test_final_runner_writes_threshold_peak_csv(tmp_path):
    protocol = _small_protocol()
    run_final_market_calibration(
        output_dir=tmp_path,
        protocol=protocol,
        baseline=_small_baseline(),
    )
    rows = (tmp_path / "threshold_peak_cids.csv").read_text().strip().splitlines()
    assert rows[0] == "replication_id,peak_cid"
    assert len(rows) == protocol.n_threshold_replications + 1


def test_final_runner_second_call_is_fully_resumable(tmp_path):
    protocol = _small_protocol()
    baseline = _small_baseline()
    first = run_final_market_calibration(output_dir=tmp_path, protocol=protocol, baseline=baseline)
    events = []
    second = run_final_market_calibration(
        output_dir=tmp_path,
        protocol=protocol,
        baseline=baseline,
        resume=True,
        progress=lambda stage, completed, total, reused: events.append((stage, reused)),
    )
    assert first.calibration == second.calibration
    assert len(events) == protocol.n_scale_replications + protocol.n_threshold_replications
    assert all(reused for _, reused in events)


def test_production_cli_help_runs_from_repo_root_without_pythonpath():
    completed = subprocess.run(
        [sys.executable, "scripts/run_refined_market_calibration.py", "--help"],
        cwd=REPO_ROOT,
        env={key: value for key, value in __import__("os").environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--stage" in completed.stdout


def test_small_protocol_expected_points_are_six():
    assert _small_protocol().expected_rolling_points_per_run == 6


def test_final_threshold_is_finite_and_positive(tmp_path):
    result = run_final_market_calibration(
        output_dir=tmp_path,
        protocol=_small_protocol(),
        baseline=_small_baseline(),
    )
    assert np.isfinite(result.calibration.cid_threshold)
    assert result.calibration.cid_threshold > 0.0
