from dataclasses import replace

import pytest

from src.experiments.refined.confirmatory_runner import ConfirmatoryTreatmentRecord
from src.experiments.refined.gamma_sweep_analysis import (
    GammaSweepTreatmentRecord,
    analyse_gamma_sweep_records,
)
from src.experiments.refined.gamma_sweep_protocol import GammaSweepProtocol


def _protocol():
    return GammaSweepProtocol(
        experiment_seed=96301,
        gamma_grid=(0.0, 0.9, 0.99, 0.999),
        n_replications=2,
        bootstrap_seed=96302,
        n_bootstrap=1000,
    )


def _record(seed: int, replication_id: int, gamma_R: float, topology: str):
    offsets = {"R": 1.0, "SW": 0.0, "SF": 2.0}
    level = 1.0 + gamma_R + offsets[topology] + 0.1 * replication_id
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
        threshold_exceeding=(gamma_R >= 0.99 and topology == "SF"),
        cid_exceedance_duration_share=0.01,
        stabilised=True,
        stabilisation_period=3,
        right_censored=False,
        mean_pairwise_action_covariance={"R": 0.1, "SW": -0.1, "SF": 0.2}[topology],
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


def _records(protocol=None):
    protocol = protocol or _protocol()
    return tuple(
        _record(protocol.experiment_seed, replication_id, gamma_R, topology)
        for replication_id in range(protocol.n_replications)
        for gamma_R in protocol.gamma_grid
        for topology in protocol.topology_labels
    )


def test_gamma_sweep_means_pairwise_signs_and_crn_flag_are_correct():
    result = analyse_gamma_sweep_records(_records(), protocol=_protocol())
    assert result.cross_gamma_common_random_numbers_verified is True
    means = {
        (item.gamma_R, item.outcome, item.topology): item.estimate
        for item in result.topology_means
    }
    assert means[(0.9, "return_volatility", "SW")] == pytest.approx(1.95)
    assert means[(0.9, "return_volatility", "R")] == pytest.approx(2.95)
    assert means[(0.9, "return_volatility", "SF")] == pytest.approx(3.95)

    contrasts = {
        (item.gamma_R, item.outcome, item.topology_left, item.topology_right): item.estimate
        for item in result.pairwise_contrasts
    }
    assert contrasts[(0.9, "return_volatility", "R", "SW")] == pytest.approx(1.0)
    assert contrasts[(0.9, "return_volatility", "R", "SF")] == pytest.approx(-1.0)
    assert contrasts[(0.9, "return_volatility", "SW", "SF")] == pytest.approx(-2.0)


def test_gamma_specific_reputation_diagnostics_are_analysed_as_relative_mechanisms():
    result = analyse_gamma_sweep_records(_records(), protocol=_protocol())
    item = next(
        x for x in result.topology_gaps
        if x.gamma_R == 0.99 and x.outcome == "mean_raw_local_reputation_std_over_sigma0"
    )
    assert item.family == "mechanism"
    assert item.relative_gap is not None
    assert item.relative_gap_ci_lower is not None
    assert item.relative_gap_ci_upper is not None


def test_gamma_sweep_bootstrap_is_reproducible():
    first = analyse_gamma_sweep_records(_records(), protocol=_protocol())
    second = analyse_gamma_sweep_records(_records(), protocol=_protocol())
    assert first.topology_means[10].ci_lower == second.topology_means[10].ci_lower
    assert first.pairwise_contrasts[20].ci_upper == second.pairwise_contrasts[20].ci_upper


def test_binary_and_signed_outcomes_do_not_get_relative_effects():
    result = analyse_gamma_sweep_records(_records(), protocol=_protocol())
    binary = next(
        item for item in result.pairwise_contrasts
        if item.gamma_R == 0.99 and item.outcome == "threshold_exceeding"
    )
    signed = next(
        item for item in result.pairwise_contrasts
        if item.gamma_R == 0.99 and item.outcome == "mean_pairwise_action_covariance"
    )
    assert binary.relative_effect is None
    assert signed.relative_effect is None


def test_incomplete_gamma_block_is_rejected():
    records = list(_records())
    records.pop()
    with pytest.raises(ValueError, match="complete R/SW/SF triplet"):
        analyse_gamma_sweep_records(records, protocol=_protocol())


def test_cross_gamma_seed_mismatch_is_rejected():
    records = list(_records())
    target = next(
        i for i, record in enumerate(records)
        if record.treatment.replication_id == 0
        and record.gamma_R == 0.99
        and record.treatment.topology_label == "SF"
    )
    records[target] = replace(
        records[target],
        treatment=replace(records[target].treatment, shock_seed=99999),
    )
    with pytest.raises(ValueError, match="cross-gamma shock-seed mismatch"):
        analyse_gamma_sweep_records(records, protocol=_protocol())


def test_wrong_alpha_anchor_is_rejected():
    records = list(_records())
    records[0] = replace(
        records[0],
        treatment=replace(records[0].treatment, alpha=0.75),
    )
    with pytest.raises(ValueError, match="alpha"):
        analyse_gamma_sweep_records(records, protocol=_protocol())
