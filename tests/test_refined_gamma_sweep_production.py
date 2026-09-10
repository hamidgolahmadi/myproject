from dataclasses import replace
import json

import pytest

import src.experiments.refined.gamma_sweep_production as production
from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.confirmatory_runner import ConfirmatoryTreatmentRecord
from src.experiments.refined.gamma_sweep_analysis import GammaSweepTreatmentRecord
from src.experiments.refined.gamma_sweep_protocol import GammaSweepProtocol
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
        scale_calibration_seed=96401,
        threshold_calibration_seed=96402,
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


def _protocol(n_replications=2):
    return GammaSweepProtocol(
        experiment_seed=96501,
        gamma_grid=(0.0, 0.9, 0.99, 0.999),
        n_replications=n_replications,
        bootstrap_seed=96502,
        n_bootstrap=1000,
    )


def _wrapped(seed: int, replication_id: int, gamma_R: float, topology: str):
    offset = {"R": 1.0, "SW": 0.0, "SF": 2.0}[topology]
    level = 1.0 + gamma_R + offset + 0.1 * replication_id
    graph_offset = {"R": 1, "SW": 2, "SF": 3}[topology]
    treatment = ConfirmatoryTreatmentRecord(
        experiment_seed=seed,
        replication_id=replication_id,
        regime="gamma_sweep",
        alpha=0.85,
        topology_label=topology,
        graph_seed=1000 + 10 * replication_id + graph_offset,
        shock_seed=2000 + replication_id,
        initial_state_seed=3000 + replication_id,
        economic_path_fingerprint=f"{gamma_R}-{topology}-{replication_id}".ljust(64, "0"),
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
        mean_hub_influence_share=0.2 * level,
        mean_attention_overlap=0.05 * level,
        mean_attention_mobility=0.01,
    )
    raw = 0.001 * level
    return GammaSweepTreatmentRecord(
        gamma_R=gamma_R,
        mean_raw_local_reputation_std=raw,
        mean_raw_local_reputation_std_over_sigma0=raw / 0.0005,
        treatment=treatment,
    )


def _fake_runner(**kwargs):
    baseline = kwargs["baseline"]
    gamma_R = float(baseline.parameters.gamma_R)
    return tuple(
        _wrapped(kwargs["experiment_seed"], kwargs["replication_id"], gamma_R, topology)
        for topology in ("R", "SW", "SF")
    )


def test_gamma_sweep_configuration_fingerprint_is_deterministic_and_sensitive():
    protocol = _protocol()
    baseline = _baseline()
    calibration = _calibration()
    first = production.gamma_sweep_configuration_fingerprint(protocol, baseline, calibration)
    second = production.gamma_sweep_configuration_fingerprint(protocol, baseline, calibration)
    changed = production.gamma_sweep_configuration_fingerprint(
        replace(protocol, n_bootstrap=2000), baseline, calibration
    )
    assert first == second
    assert len(first) == 64
    assert changed != first


def test_range_writes_one_triplet_checkpoint_for_gamma_and_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_gamma_replication", _fake_runner)
    records = production.run_gamma_sweep_range(
        gamma_index=2,
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert len(records) == 6
    for replication_id in (0, 1):
        path = tmp_path / "checkpoints" / "gamma_02" / f"replication_{replication_id:04d}.json"
        payload = json.loads(path.read_text())
        assert payload["gamma_R"] == 0.99
        assert payload["status"] == "complete"
        assert len(payload["records"]) == 3
        assert all(item["gamma_R"] == 0.99 for item in payload["records"])
        assert all(item["alpha"] == 0.85 for item in payload["records"])
        assert all("mean_raw_local_reputation_std" in item for item in payload["records"])


def test_resume_reuses_valid_gamma_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_gamma_replication", _fake_runner)
    production.run_gamma_sweep_range(
        gamma_index=1,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )

    def forbidden(**kwargs):
        raise AssertionError("valid D048 checkpoint should have been reused")

    monkeypatch.setattr(production, "run_paired_gamma_replication", forbidden)
    records = production.run_gamma_sweep_range(
        gamma_index=1,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
        resume=True,
    )
    assert len(records) == 3


def test_stale_gamma_checkpoint_configuration_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_gamma_replication", _fake_runner)
    production.run_gamma_sweep_range(
        gamma_index=1,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        production.run_gamma_sweep_range(
            gamma_index=1,
            start_replication=0,
            stop_replication=1,
            output_dir=tmp_path,
            protocol=replace(_protocol(), n_bootstrap=2000),
            baseline=_baseline(),
            calibration=_calibration(),
            resume=True,
        )


def test_full_loader_refuses_missing_gamma_replication_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_gamma_replication", _fake_runner)
    production.run_gamma_sweep_range(
        gamma_index=0,
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="missing"):
        production.load_all_gamma_sweep_records(
            output_dir=tmp_path,
            protocol=_protocol(),
            baseline=_baseline(),
            calibration=_calibration(),
        )


def test_finalize_writes_complete_exploratory_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_gamma_replication", _fake_runner)
    protocol = _protocol()
    for gamma_index in range(protocol.n_gamma):
        production.run_gamma_sweep_range(
            gamma_index=gamma_index,
            start_replication=0,
            stop_replication=protocol.n_replications,
            output_dir=tmp_path,
            protocol=protocol,
            baseline=_baseline(),
            calibration=_calibration(),
        )
    paths = production.finalize_gamma_sweep_production(
        output_dir=tmp_path,
        protocol=protocol,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert set(paths) == {"records", "metadata", "analysis", "means", "gaps", "contrasts"}
    assert all(path.exists() for path in paths.values())
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["final_gamma_sweep"] is True
    assert metadata["confirmatory"] is False
    assert metadata["n_gamma"] == 4
    assert metadata["n_treatment_records"] == 24
    assert metadata["d046_selected_alpha_anchor"] == 0.85
    assert metadata["d047_selected_beta_anchor"] == 5.0
    assert metadata["gamma_zero_control"].startswith("gamma_R=0")


def test_invalid_gamma_index_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="gamma_index"):
        production.run_gamma_sweep_range(
            gamma_index=99,
            start_replication=0,
            stop_replication=1,
            output_dir=tmp_path,
            protocol=_protocol(),
            baseline=_baseline(),
            calibration=_calibration(),
        )
