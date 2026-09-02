from dataclasses import fields, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.experiments.refined.baseline_specification import (
    first_refined_baseline_specification,
)
from src.experiments.refined.calibration_smoke import (
    NoSocialCalibrationSmokeProtocol,
    _no_social_parameters,
    run_no_social_calibration_smoke,
    write_no_social_calibration_smoke,
)
from src.experiments.refined.cid import standardise_cid_components
from src.experiments.refined.market_calibration import (
    first_market_evaluation_calibration_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_baseline():
    base = first_refined_baseline_specification()
    return replace(base, n_agents=8, k=2, hub_q=2, p_sw=0.25)


@pytest.fixture(scope="module")
def smoke_result():
    return run_no_social_calibration_smoke(
        smoke_protocol=NoSocialCalibrationSmokeProtocol(
            scale_seed=1234501,
            threshold_seed=1234502,
            n_scale_replications=2,
            n_threshold_replications=2,
        ),
        baseline=_small_baseline(),
    )


def test_direct_calibration_smoke_script_can_import_project_package():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_refined_no_social_calibration_smoke.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "D042 calibration smoke" in completed.stdout


def test_default_smoke_protocol_values():
    protocol = NoSocialCalibrationSmokeProtocol()
    assert protocol.scale_seed == 2026090204
    assert protocol.threshold_seed == 2026090205
    assert protocol.n_scale_replications == 3
    assert protocol.n_threshold_replications == 3


def test_smoke_protocol_is_seed_disjoint_from_final_d042():
    smoke = NoSocialCalibrationSmokeProtocol()
    final = first_market_evaluation_calibration_protocol()
    assert {smoke.scale_seed, smoke.threshold_seed}.isdisjoint(
        {final.scale_calibration_seed, final.threshold_calibration_seed}
    )


def test_smoke_market_protocol_preserves_d042_method():
    smoke = NoSocialCalibrationSmokeProtocol(n_scale_replications=2, n_threshold_replications=4)
    method = smoke.market_protocol()
    final = first_market_evaluation_calibration_protocol()
    assert method.horizon == final.horizon
    assert method.burn_in == final.burn_in
    assert method.rolling_window == final.rolling_window
    assert method.calibration_alpha == 0.0
    assert method.cid_weights == final.cid_weights
    assert method.cid_peak_quantile == final.cid_peak_quantile
    assert method.stabilisation_length == final.stabilisation_length
    assert method.n_scale_replications == 2
    assert method.n_threshold_replications == 4


@pytest.mark.parametrize("forbidden", [2026090201, 2026090202])
def test_smoke_protocol_rejects_final_d042_seed_reuse(forbidden):
    with pytest.raises(ValueError):
        NoSocialCalibrationSmokeProtocol(scale_seed=forbidden)


def test_smoke_protocol_rejects_equal_sample_seeds():
    with pytest.raises(ValueError):
        NoSocialCalibrationSmokeProtocol(scale_seed=77, threshold_seed=77)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_smoke_protocol_rejects_invalid_scale_replication_count(value):
    with pytest.raises((TypeError, ValueError)):
        NoSocialCalibrationSmokeProtocol(n_scale_replications=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_smoke_protocol_rejects_invalid_threshold_replication_count(value):
    with pytest.raises((TypeError, ValueError)):
        NoSocialCalibrationSmokeProtocol(n_threshold_replications=value)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_smoke_protocol_rejects_invalid_scale_seed(value):
    with pytest.raises((TypeError, ValueError)):
        NoSocialCalibrationSmokeProtocol(scale_seed=value)


def test_no_social_parameters_change_only_alpha():
    baseline = first_refined_baseline_specification()
    no_social = _no_social_parameters(baseline)
    assert no_social.alpha == 0.0
    for field in fields(no_social):
        if field.name != "alpha":
            assert getattr(no_social, field.name) == getattr(baseline.parameters, field.name)


def test_final_d042_protocol_remains_500_plus_500():
    final = first_market_evaluation_calibration_protocol()
    assert final.n_scale_replications == 500
    assert final.n_threshold_replications == 500


def test_smoke_result_contains_requested_replication_counts(smoke_result):
    assert len(smoke_result.scale_paths) == 2
    assert len(smoke_result.threshold_paths) == 2
    assert len(smoke_result.threshold_peak_cids) == 2


def test_smoke_paths_have_exactly_951_rolling_points(smoke_result):
    assert smoke_result.market_protocol.expected_rolling_points_per_run == 951
    assert all(len(path) == 951 for path in smoke_result.scale_paths)
    assert all(len(path) == 951 for path in smoke_result.threshold_paths)


def test_smoke_rolling_endpoint_alignment(smoke_result):
    for path in smoke_result.scale_paths + smoke_result.threshold_paths:
        assert path[0].endpoint_period == 50
        assert path[0].window_start_period == 1
        assert path[-1].endpoint_period == 1000
        assert path[-1].window_start_period == 951


def test_smoke_reference_scales_are_strictly_positive(smoke_result):
    scales = smoke_result.calibration.reference_scales
    assert scales.return_scale > 0.0
    assert scales.belief_scale > 0.0
    assert scales.order_flow_scale > 0.0


def test_smoke_cid_threshold_is_finite_and_positive(smoke_result):
    threshold = smoke_result.calibration.cid_threshold
    assert np.isfinite(threshold)
    assert threshold > 0.0


def test_smoke_threshold_peaks_are_finite_and_positive(smoke_result):
    peaks = np.asarray(smoke_result.threshold_peak_cids)
    assert np.all(np.isfinite(peaks))
    assert np.all(peaks > 0.0)


def test_smoke_standardised_cid_paths_are_finite(smoke_result):
    scales = smoke_result.calibration.reference_scales
    weights = smoke_result.calibration.cid_weights
    for components in smoke_result.threshold_paths:
        cid_path = standardise_cid_components(components, scales=scales, weights=weights)
        assert len(cid_path) == 951
        assert all(np.isfinite(point.cid) and point.cid >= 0.0 for point in cid_path)


def test_smoke_calibration_uses_equal_frozen_weights(smoke_result):
    assert smoke_result.calibration.cid_weights == first_market_evaluation_calibration_protocol().cid_weights


def test_smoke_calibration_has_no_component_guardrails(smoke_result):
    calibration = smoke_result.calibration
    assert calibration.return_guardrail is None
    assert calibration.belief_guardrail is None
    assert calibration.order_flow_guardrail is None


def test_smoke_rejects_baseline_horizon_mismatch_before_simulation():
    baseline = replace(_small_baseline(), horizon=999)
    with pytest.raises(ValueError, match="horizon"):
        run_no_social_calibration_smoke(
            smoke_protocol=NoSocialCalibrationSmokeProtocol(
                scale_seed=7701,
                threshold_seed=7702,
                n_scale_replications=1,
                n_threshold_replications=1,
            ),
            baseline=baseline,
        )


def test_smoke_rejects_wrong_protocol_type():
    with pytest.raises(TypeError):
        run_no_social_calibration_smoke(smoke_protocol=object())


def test_smoke_rejects_wrong_baseline_type():
    with pytest.raises(TypeError):
        run_no_social_calibration_smoke(baseline=object())


def test_smoke_writer_persists_json_artifact(tmp_path, smoke_result):
    path = write_no_social_calibration_smoke(smoke_result, outdir=tmp_path)
    assert path == tmp_path / "calibration_smoke.json"
    assert path.exists()


def test_smoke_writer_labels_artifact_as_not_final(tmp_path, smoke_result):
    path = write_no_social_calibration_smoke(smoke_result, outdir=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["final_calibration"] is False
    assert "not final" in payload["purpose"].lower()


def test_smoke_writer_records_alpha_and_rolling_count(tmp_path, smoke_result):
    path = write_no_social_calibration_smoke(smoke_result, outdir=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["calibration_alpha"] == 0.0
    assert payload["expected_rolling_points_per_run"] == 951


def test_smoke_writer_records_frozen_sigma0(tmp_path, smoke_result):
    path = write_no_social_calibration_smoke(smoke_result, outdir=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["frozen_baseline_sigma_0"] == pytest.approx(5e-4)


def test_smoke_writer_records_positive_calibration_values(tmp_path, smoke_result):
    path = write_no_social_calibration_smoke(smoke_result, outdir=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["reference_scales"]["c_ret"] > 0.0
    assert payload["reference_scales"]["c_bel"] > 0.0
    assert payload["reference_scales"]["c_F"] > 0.0
    assert payload["c_CID"] > 0.0


def test_smoke_writer_rejects_wrong_result_type(tmp_path):
    with pytest.raises(TypeError):
        write_no_social_calibration_smoke(object(), outdir=tmp_path)
