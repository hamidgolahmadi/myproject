from dataclasses import replace

from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.confirmatory_runner import run_paired_confirmatory_replication
from src.experiments.refined.joint_interaction_protocol import JointCell
from src.experiments.refined.joint_interaction_runner import run_paired_joint_replication
from src.experiments.refined.market_calibration import (
    MarketEvaluationCalibration,
    MarketEvaluationCalibrationProtocol,
)


def _baseline():
    return replace(
        first_refined_baseline_specification(),
        n_agents=8,
        k=2,
        horizon=8,
        hub_q=2,
        p_sw=0.25,
    )


def _calibration():
    protocol = MarketEvaluationCalibrationProtocol(
        scale_calibration_seed=97101,
        threshold_calibration_seed=97102,
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


def test_joint_runner_records_exact_cell_metadata_and_triplet():
    cell = JointCell("T", "factorial", 0.85, 5.0, 0.99)
    records = run_paired_joint_replication(
        experiment_seed=97201,
        replication_id=3,
        cell=cell,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert tuple(item.treatment.topology_label for item in records) == ("R", "SW", "SF")
    assert all(item.cell_id == "T" for item in records)
    assert all(item.alpha == 0.85 and item.beta == 5.0 and item.gamma_R == 0.99 for item in records)
    assert all(item.treatment.regime == "joint_interaction" for item in records)
    assert all(item.mean_raw_local_reputation_std_over_sigma0 >= 0.0 for item in records)


def test_joint_alpha_zero_control_has_exact_topology_null_economic_path():
    cell = JointCell("C", "alpha0_control", 0.0, 5.0, 0.9)
    records = run_paired_joint_replication(
        experiment_seed=97202,
        replication_id=4,
        cell=cell,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert len({item.treatment.economic_path_fingerprint for item in records}) == 1


def test_joint_d043_anchor_preserves_confirmatory_treatment_outputs():
    baseline = _baseline()
    calibration = _calibration()
    cell = JointCell("A", "d043_anchor", 0.75, 1.0, 0.9)
    wrapped = run_paired_joint_replication(
        experiment_seed=97203,
        replication_id=5,
        cell=cell,
        baseline=baseline,
        calibration=calibration,
    )
    reference = run_paired_confirmatory_replication(
        experiment_seed=97203,
        replication_id=5,
        regime="joint_interaction",
        baseline=baseline,
        calibration=calibration,
    )
    assert tuple(item.treatment for item in wrapped) == reference
