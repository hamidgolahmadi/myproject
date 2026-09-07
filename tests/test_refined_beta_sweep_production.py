from dataclasses import replace
import json

import pytest

import src.experiments.refined.beta_sweep_production as production
from src.experiments.refined.beta_sweep_protocol import BetaSweepProtocol
from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.confirmatory_runner import ConfirmatoryTreatmentRecord
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
        scale_calibration_seed=95101,
        threshold_calibration_seed=95102,
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
    return BetaSweepProtocol(
        experiment_seed=95201,
        alpha_anchor=0.85,
        beta_grid=(0.0, 1.0, 2.0, 5.0, 1000.0),
        n_replications=n_replications,
        bootstrap_seed=95202,
        n_bootstrap=1000,
    )


def _record(seed: int, replication_id: int, alpha: float, beta: float, topology: str):
    offset = {"R": 1.0, "SW": 0.0, "SF": 2.0}[topology]
    level = 1.0 + 0.01 * beta + offset + 0.1 * replication_id
    graph_offset = {"R": 1, "SW": 2, "SF": 3}[topology]
    return ConfirmatoryTreatmentRecord(
        experiment_seed=seed,
        replication_id=replication_id,
        regime="beta_sweep",
        alpha=alpha,
        topology_label=topology,
        graph_seed=1000 + 10 * replication_id + graph_offset,
        shock_seed=2000 + replication_id,
        initial_state_seed=3000 + replication_id,
        economic_path_fingerprint=f"{beta}-{topology}-{replication_id}".ljust(64, "0"),
        return_volatility=level,
        rms_mispricing=level,
        maximum_absolute_mispricing=level,
        mean_absolute_order_flow_per_agent=level,
        mean_absolute_return=level,
        time_averaged_belief_variance=level,
        peak_cid=level,
        threshold_exceeding=(beta >= 5.0 and topology == "SF"),
        cid_exceedance_duration_share=0.1,
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


def _fake_runner(**kwargs):
    baseline = kwargs["baseline"]
    beta = float(baseline.parameters.beta)
    alpha = float(baseline.parameters.alpha)
    return tuple(
        _record(kwargs["experiment_seed"], kwargs["replication_id"], alpha, beta, topology)
        for topology in ("R", "SW", "SF")
    )


def test_beta_sweep_configuration_fingerprint_is_deterministic_and_sensitive():
    protocol = _protocol()
    baseline = _baseline()
    calibration = _calibration()
    first = production.beta_sweep_configuration_fingerprint(protocol, baseline, calibration)
    second = production.beta_sweep_configuration_fingerprint(protocol, baseline, calibration)
    changed = production.beta_sweep_configuration_fingerprint(
        replace(protocol, n_bootstrap=2000), baseline, calibration
    )
    assert first == second
    assert len(first) == 64
    assert changed != first


def test_range_writes_one_triplet_checkpoint_for_beta_and_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    records = production.run_beta_sweep_range(
        beta_index=3,
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert len(records) == 6
    for replication_id in (0, 1):
        path = tmp_path / "checkpoints" / "beta_03" / f"replication_{replication_id:04d}.json"
        payload = json.loads(path.read_text())
        assert payload["beta"] == 5.0
        assert payload["status"] == "complete"
        assert len(payload["records"]) == 3
        assert all(item["beta"] == 5.0 for item in payload["records"])
        assert all(item["alpha"] == 0.85 for item in payload["records"])


def test_resume_reuses_valid_beta_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    production.run_beta_sweep_range(
        beta_index=2,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )

    def forbidden(**kwargs):
        raise AssertionError("valid D047 checkpoint should have been reused")

    monkeypatch.setattr(production, "run_paired_confirmatory_replication", forbidden)
    records = production.run_beta_sweep_range(
        beta_index=2,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
        resume=True,
    )
    assert len(records) == 3


def test_stale_beta_checkpoint_configuration_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    production.run_beta_sweep_range(
        beta_index=1,
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        production.run_beta_sweep_range(
            beta_index=1,
            start_replication=0,
            stop_replication=1,
            output_dir=tmp_path,
            protocol=replace(_protocol(), n_bootstrap=2000),
            baseline=_baseline(),
            calibration=_calibration(),
            resume=True,
        )


def test_full_loader_refuses_missing_beta_replication_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    production.run_beta_sweep_range(
        beta_index=0,
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="missing"):
        production.load_all_beta_sweep_records(
            output_dir=tmp_path,
            protocol=_protocol(),
            baseline=_baseline(),
            calibration=_calibration(),
        )


def test_finalize_writes_complete_exploratory_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    protocol = _protocol()
    for beta_index in range(protocol.n_beta):
        production.run_beta_sweep_range(
            beta_index=beta_index,
            start_replication=0,
            stop_replication=protocol.n_replications,
            output_dir=tmp_path,
            protocol=protocol,
            baseline=_baseline(),
            calibration=_calibration(),
        )
    paths = production.finalize_beta_sweep_production(
        output_dir=tmp_path,
        protocol=protocol,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert set(paths) == {"records", "metadata", "analysis", "means", "gaps", "contrasts"}
    assert all(path.exists() for path in paths.values())
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["final_beta_sweep"] is True
    assert metadata["confirmatory"] is False
    assert metadata["n_beta"] == 5
    assert metadata["n_treatment_records"] == 30
    assert metadata["d046_selected_alpha_anchor"] == 0.85


def test_invalid_beta_index_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="beta_index"):
        production.run_beta_sweep_range(
            beta_index=99,
            start_replication=0,
            stop_replication=1,
            output_dir=tmp_path,
            protocol=_protocol(),
            baseline=_baseline(),
            calibration=_calibration(),
        )
