from dataclasses import replace

from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.confirmatory_runner import run_paired_confirmatory_replication
from src.experiments.refined.gamma_sweep_runner import run_paired_gamma_replication
from src.experiments.refined.market_calibration import (
    MarketEvaluationCalibration,
    MarketEvaluationCalibrationProtocol,
)


def _baseline():
    baseline = replace(
        first_refined_baseline_specification(),
        n_agents=8,
        k=2,
        horizon=8,
        hub_q=2,
        p_sw=0.25,
    )
    parameters = replace(baseline.parameters, alpha=0.85, beta=5.0, gamma_R=0.9)
    return replace(baseline, parameters=parameters)


def _calibration():
    protocol = MarketEvaluationCalibrationProtocol(
        scale_calibration_seed=96101,
        threshold_calibration_seed=96102,
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


def test_gamma_runner_preserves_frozen_confirmatory_treatment_outputs_exactly():
    baseline = _baseline()
    calibration = _calibration()
    wrapped = run_paired_gamma_replication(
        experiment_seed=96201,
        replication_id=3,
        baseline=baseline,
        calibration=calibration,
    )
    reference = run_paired_confirmatory_replication(
        experiment_seed=96201,
        replication_id=3,
        regime="gamma_sweep",
        baseline=baseline,
        calibration=calibration,
    )
    assert tuple(item.treatment for item in wrapped) == reference
    assert tuple(item.gamma_R for item in wrapped) == (0.9, 0.9, 0.9)
    assert all(item.mean_raw_local_reputation_std >= 0.0 for item in wrapped)
    assert all(item.mean_raw_local_reputation_std_over_sigma0 >= 0.0 for item in wrapped)
    for item in wrapped:
        assert item.mean_raw_local_reputation_std_over_sigma0 == (
            item.mean_raw_local_reputation_std / baseline.parameters.sigma_0
        )
