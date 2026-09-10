from dataclasses import replace
import json

import pytest

import src.experiments.refined.joint_interaction_production as production
from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.confirmatory_runner import ConfirmatoryTreatmentRecord
from src.experiments.refined.joint_interaction_analysis import JointTreatmentRecord
from src.experiments.refined.joint_interaction_protocol import JointInteractionProtocol
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
    p = MarketEvaluationCalibrationProtocol(
        scale_calibration_seed=97401,
        threshold_calibration_seed=97402,
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
        protocol=p,
        reference_scales=CIDReferenceScales(
            return_scale=0.01,
            belief_scale=0.01,
            order_flow_scale=0.2,
        ),
        cid_weights=p.cid_weights,
        cid_threshold=2.0,
    )


def _protocol(n_replications=2):
    return JointInteractionProtocol(
        experiment_seed=97501,
        bootstrap_seed=97502,
        n_replications=n_replications,
        n_bootstrap=1000,
    )


def _wrapped(seed, replication_id, cell, topology):
    d = cell.alpha * cell.beta * cell.gamma_R
    level = 10.0 + 0.1 * replication_id + {"SW": 0.0, "R": 0.5 * d, "SF": d}[topology]
    graph_seed = 1000 + 10 * replication_id + {"R": 1, "SW": 2, "SF": 3}[topology]
    fingerprint = (
        f"alpha0-{replication_id}".ljust(64, "0")
        if cell.cell_kind == "alpha0_control"
        else f"{cell.cell_id}-{topology}-{replication_id}".ljust(64, "0")
    )
    treatment = ConfirmatoryTreatmentRecord(
        experiment_seed=seed,
        replication_id=replication_id,
        regime="joint_interaction",
        alpha=cell.alpha,
        topology_label=topology,
        graph_seed=graph_seed,
        shock_seed=2000 + replication_id,
        initial_state_seed=3000 + replication_id,
        economic_path_fingerprint=fingerprint,
        return_volatility=level,
        rms_mispricing=level,
        maximum_absolute_mispricing=level,
        mean_absolute_order_flow_per_agent=level,
        mean_absolute_return=level,
        time_averaged_belief_variance=level,
        peak_cid=level,
        threshold_exceeding=False,
        cid_exceedance_duration_share=0.0,
        stabilised=True,
        stabilisation_period=3,
        right_censored=False,
        mean_pairwise_action_covariance=0.01 * level,
        mean_sum_individual_action_variances=level,
        mean_aggregate_order_flow_variance=10.0 * level,
        in_degree_gini=0.1,
        hub_link_share=0.1,
        global_clustering=0.1,
        average_path_length_lcc=2.0,
        largest_component_share=1.0,
        mean_attention_entropy=0.5,
        mean_effective_sources=2.0,
        mean_influence_hhi=0.1,
        mean_hub_influence_share=0.02 * level,
        mean_attention_overlap=0.005 * level,
        mean_attention_mobility=0.01,
    )
    raw = 0.001 * (1.0 + cell.gamma_R)
    return JointTreatmentRecord(
        cell_id=cell.cell_id,
        cell_kind=cell.cell_kind,
        alpha=cell.alpha,
        beta=cell.beta,
        gamma_R=cell.gamma_R,
        mean_raw_local_reputation_std=raw,
        mean_raw_local_reputation_std_over_sigma0=raw / 0.0005,
        treatment=treatment,
    )


def _fake_runner(**kwargs):
    cell = kwargs["cell"]
    return tuple(
        _wrapped(kwargs["experiment_seed"], kwargs["replication_id"], cell, topology)
        for topology in ("R", "SW", "SF")
    )


def test_joint_configuration_fingerprint_is_deterministic_and_sensitive():
    p = _protocol()
    baseline = _baseline()
    calibration = _calibration()
    first = production.joint_interaction_configuration_fingerprint(p, baseline, calibration)
    second = production.joint_interaction_configuration_fingerprint(p, baseline, calibration)
    changed = production.joint_interaction_configuration_fingerprint(
        replace(p, n_bootstrap=2000), baseline, calibration
    )
    assert first == second
    assert len(first) == 64
    assert changed != first


def test_joint_range_writes_one_triplet_checkpoint_per_cell_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_joint_replication", _fake_runner)
    p = _protocol()
    records = production.run_joint_interaction_range(
        cell_index=17,
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=p,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert len(records) == 6
    cell = p.cells[17]
    for replication_id in (0, 1):
        path = tmp_path / "checkpoints" / "cell_17" / f"replication_{replication_id:04d}.json"
        payload = json.loads(path.read_text())
        assert payload["status"] == "complete"
        assert payload["cell_id"] == cell.cell_id
        assert payload["alpha"] == cell.alpha
        assert payload["beta"] == cell.beta
        assert payload["gamma_R"] == cell.gamma_R
        assert len(payload["records"]) == 3


def test_joint_resume_reuses_valid_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_joint_replication", _fake_runner)
    p = _protocol()
    production.run_joint_interaction_range(
        cell_index=3,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=p,
        baseline=_baseline(),
        calibration=_calibration(),
    )

    def forbidden(**kwargs):
        raise AssertionError("valid D049 checkpoint should have been reused")

    monkeypatch.setattr(production, "run_paired_joint_replication", forbidden)
    records = production.run_joint_interaction_range(
        cell_index=3,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=p,
        baseline=_baseline(),
        calibration=_calibration(),
        resume=True,
    )
    assert len(records) == 3


def test_joint_stale_checkpoint_configuration_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_joint_replication", _fake_runner)
    p = _protocol()
    production.run_joint_interaction_range(
        cell_index=4,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=p,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        production.run_joint_interaction_range(
            cell_index=4,
            start_replication=0,
            stop_replication=1,
            output_dir=tmp_path,
            protocol=replace(p, n_bootstrap=2000),
            baseline=_baseline(),
            calibration=_calibration(),
            resume=True,
        )


def test_joint_full_loader_refuses_missing_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_joint_replication", _fake_runner)
    p = _protocol()
    production.run_joint_interaction_range(
        cell_index=0,
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=p,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="missing"):
        production.load_all_joint_interaction_records(
            output_dir=tmp_path,
            protocol=p,
            baseline=_baseline(),
            calibration=_calibration(),
        )


def test_joint_finalize_writes_all_final_artifacts_and_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_joint_replication", _fake_runner)
    p = _protocol()
    baseline = _baseline()
    calibration = _calibration()
    for cell_index in range(p.n_cells):
        production.run_joint_interaction_range(
            cell_index=cell_index,
            start_replication=0,
            stop_replication=p.n_replications,
            output_dir=tmp_path,
            protocol=p,
            baseline=baseline,
            calibration=calibration,
        )
    paths = production.finalize_joint_interaction_production(
        output_dir=tmp_path,
        protocol=p,
        baseline=baseline,
        calibration=calibration,
    )
    assert set(paths) == {
        "records", "metadata", "analysis", "means", "gaps", "contrasts", "interactions"
    }
    assert all(path.exists() for path in paths.values())
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["final_joint_interaction"] is True
    assert metadata["confirmatory"] is False
    assert metadata["n_cells"] == 50
    assert metadata["n_factorial_cells"] == 48
    assert metadata["n_checkpoints"] == 100
    assert metadata["n_treatment_records"] == 300
    assert metadata["alpha_zero_economic_path_null_verified"] is True
    assert metadata["cross_cell_common_random_numbers_verified"] is True
