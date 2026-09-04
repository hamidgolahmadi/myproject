from dataclasses import replace
import json

import pytest

import src.experiments.refined.confirmatory_production as production
from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.cid import CIDReferenceScales, CIDWeights
from src.experiments.refined.confirmatory_protocol import ConfirmatoryProductionProtocol
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
        scale_calibration_seed=93101,
        threshold_calibration_seed=93102,
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
    return ConfirmatoryProductionProtocol(
        experiment_seed=93001,
        n_replications=n_replications,
        bootstrap_seed=93002,
        n_bootstrap=1000,
    )


def _record(experiment_seed: int, replication_id: int, topology: str):
    level = {"R": 1.0, "SW": 2.0, "SF": 3.0}[topology] + 0.1 * replication_id
    return ConfirmatoryTreatmentRecord(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        regime="baseline",
        alpha=0.75,
        topology_label=topology,
        graph_seed=100 + replication_id,
        shock_seed=200 + replication_id,
        initial_state_seed=300 + replication_id,
        economic_path_fingerprint=(topology + str(replication_id)).ljust(64, "0"),
        return_volatility=level,
        rms_mispricing=level,
        maximum_absolute_mispricing=level,
        mean_absolute_order_flow_per_agent=level,
        mean_absolute_return=level,
        time_averaged_belief_variance=level,
        peak_cid=level,
        threshold_exceeding=topology == "SF",
        cid_exceedance_duration_share=0.1,
        stabilised=True,
        stabilisation_period=3,
        right_censored=False,
        mean_pairwise_action_covariance={"R": -0.1, "SW": 0.0, "SF": 0.1}[topology],
        mean_sum_individual_action_variances=level,
        mean_aggregate_order_flow_variance=level,
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
    return tuple(
        _record(kwargs["experiment_seed"], kwargs["replication_id"], topology)
        for topology in ("R", "SW", "SF")
    )


def test_configuration_fingerprint_is_deterministic_and_protocol_sensitive():
    protocol = _protocol()
    baseline = _baseline()
    calibration = _calibration()
    first = production.confirmatory_configuration_fingerprint(protocol, baseline, calibration)
    second = production.confirmatory_configuration_fingerprint(protocol, baseline, calibration)
    changed = production.confirmatory_configuration_fingerprint(
        replace(protocol, n_bootstrap=2000), baseline, calibration
    )
    assert first == second
    assert len(first) == 64
    assert changed != first


def test_range_run_writes_one_complete_triplet_checkpoint_per_replication(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    records = production.run_confirmatory_production_range(
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert len(records) == 6
    for replication_id in (0, 1):
        path = tmp_path / "replications" / f"replication_{replication_id:04d}.json"
        payload = json.loads(path.read_text())
        assert payload["status"] == "complete"
        assert len(payload["records"]) == 3


def test_resume_reuses_valid_triplet_checkpoint_without_recomputation(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    production.run_confirmatory_production_range(
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )

    def forbidden(**kwargs):
        raise AssertionError("valid checkpoint should have been reused")

    monkeypatch.setattr(production, "run_paired_confirmatory_replication", forbidden)
    records = production.run_confirmatory_production_range(
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
        resume=True,
    )
    assert len(records) == 3


def test_stale_checkpoint_configuration_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    production.run_confirmatory_production_range(
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        production.run_confirmatory_production_range(
            start_replication=0,
            stop_replication=1,
            output_dir=tmp_path,
            protocol=replace(_protocol(), n_bootstrap=2000),
            baseline=_baseline(),
            calibration=_calibration(),
            resume=True,
        )


def test_full_loader_refuses_missing_predeclared_replications(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    production.run_confirmatory_production_range(
        start_replication=0,
        stop_replication=1,
        output_dir=tmp_path,
        protocol=_protocol(n_replications=2),
        baseline=_baseline(),
        calibration=_calibration(),
    )
    with pytest.raises(RuntimeError, match="missing"):
        production.load_all_confirmatory_records(
            output_dir=tmp_path,
            protocol=_protocol(n_replications=2),
            baseline=_baseline(),
            calibration=_calibration(),
        )


def test_finalize_writes_only_after_complete_sample_and_marks_final(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "run_paired_confirmatory_replication", _fake_runner)
    protocol = _protocol(n_replications=2)
    production.run_confirmatory_production_range(
        start_replication=0,
        stop_replication=2,
        output_dir=tmp_path,
        protocol=protocol,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    paths = production.finalize_confirmatory_production(
        output_dir=tmp_path,
        protocol=protocol,
        baseline=_baseline(),
        calibration=_calibration(),
    )
    assert set(paths) == {"records", "metadata", "analysis", "means", "gaps", "contrasts"}
    assert all(path.exists() for path in paths.values())
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["final_confirmatory"] is True
    assert metadata["n_complete_paired_replications"] == 2
    assert metadata["n_treatment_records"] == 6
    assert "matched R/SW/SF" in metadata["bootstrap_design"]


def test_invalid_replication_range_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="0 <= start"):
        production.run_confirmatory_production_range(
            start_replication=2,
            stop_replication=2,
            output_dir=tmp_path,
            protocol=_protocol(),
            baseline=_baseline(),
            calibration=_calibration(),
        )
