from dataclasses import replace

import pytest

from src.experiments.refined.alpha_sweep_analysis import analyse_alpha_sweep_records
from src.experiments.refined.alpha_sweep_protocol import AlphaSweepProtocol
from src.experiments.refined.confirmatory_runner import ConfirmatoryTreatmentRecord


def _protocol():
    return AlphaSweepProtocol(
        experiment_seed=94001,
        alpha_grid=(0.0, 0.75, 1.0),
        n_replications=2,
        bootstrap_seed=94002,
        n_bootstrap=1000,
    )


def _record(seed: int, replication_id: int, alpha: float, topology: str):
    offsets = {"R": 1.0, "SW": 0.0, "SF": 2.0}
    if alpha == 0.0:
        level = 1.0 + 0.1 * replication_id
        signed = 0.0
        fingerprint = f"null-{replication_id}".ljust(64, "0")
    else:
        level = 1.0 + alpha + offsets[topology] + 0.1 * replication_id
        signed = {"R": 0.1, "SW": -0.1, "SF": 0.2}[topology] + 0.01 * alpha
        fingerprint = f"{alpha}-{topology}-{replication_id}".ljust(64, "0")

    return ConfirmatoryTreatmentRecord(
        experiment_seed=seed,
        replication_id=replication_id,
        regime="alpha_sweep",
        alpha=alpha,
        topology_label=topology,
        graph_seed=100 + replication_id,
        shock_seed=200 + replication_id,
        initial_state_seed=300 + replication_id,
        economic_path_fingerprint=fingerprint,
        return_volatility=level,
        rms_mispricing=level,
        maximum_absolute_mispricing=level,
        mean_absolute_order_flow_per_agent=level,
        mean_absolute_return=level,
        time_averaged_belief_variance=level,
        peak_cid=level,
        threshold_exceeding=(alpha > 0.5 and topology == "SF"),
        cid_exceedance_duration_share=0.1 * alpha,
        stabilised=True,
        stabilisation_period=3,
        right_censored=False,
        mean_pairwise_action_covariance=signed,
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


def _records(protocol=None):
    protocol = protocol or _protocol()
    return tuple(
        _record(protocol.experiment_seed, replication_id, alpha, topology)
        for replication_id in range(protocol.n_replications)
        for alpha in protocol.alpha_grid
        for topology in protocol.topology_labels
    )


def test_alpha_sweep_means_and_pairwise_signs_are_correct():
    result = analyse_alpha_sweep_records(_records(), protocol=_protocol())
    means = {
        (item.alpha, item.outcome, item.topology): item.estimate
        for item in result.topology_means
    }
    assert means[(1.0, "return_volatility", "SW")] == pytest.approx(2.05)
    assert means[(1.0, "return_volatility", "R")] == pytest.approx(3.05)
    assert means[(1.0, "return_volatility", "SF")] == pytest.approx(4.05)

    contrasts = {
        (item.alpha, item.outcome, item.topology_left, item.topology_right): item.estimate
        for item in result.pairwise_contrasts
    }
    assert contrasts[(1.0, "return_volatility", "R", "SW")] == pytest.approx(1.0)
    assert contrasts[(1.0, "return_volatility", "R", "SF")] == pytest.approx(-1.0)
    assert contrasts[(1.0, "return_volatility", "SW", "SF")] == pytest.approx(-2.0)


def test_alpha_zero_exact_topology_null_is_verified():
    result = analyse_alpha_sweep_records(_records(), protocol=_protocol())
    assert result.alpha_zero_economic_path_null_verified is True
    zero_contrasts = [
        item
        for item in result.pairwise_contrasts
        if item.alpha == 0.0 and item.outcome == "return_volatility"
    ]
    assert all(item.estimate == pytest.approx(0.0) for item in zero_contrasts)


def test_alpha_sweep_bootstrap_is_reproducible():
    first = analyse_alpha_sweep_records(_records(), protocol=_protocol())
    second = analyse_alpha_sweep_records(_records(), protocol=_protocol())
    assert first.topology_means[10].ci_lower == second.topology_means[10].ci_lower
    assert first.topology_means[10].ci_upper == second.topology_means[10].ci_upper
    assert first.pairwise_contrasts[20].ci_lower == second.pairwise_contrasts[20].ci_lower


def test_binary_and_signed_outcomes_do_not_get_relative_effects():
    result = analyse_alpha_sweep_records(_records(), protocol=_protocol())
    binary = next(
        item for item in result.pairwise_contrasts
        if item.alpha == 1.0 and item.outcome == "threshold_exceeding"
    )
    signed = next(
        item for item in result.pairwise_contrasts
        if item.alpha == 1.0 and item.outcome == "mean_pairwise_action_covariance"
    )
    assert binary.relative_effect is None
    assert signed.relative_effect is None


def test_incomplete_alpha_block_is_rejected():
    records = list(_records())
    records.pop()
    with pytest.raises(ValueError, match="complete R/SW/SF triplet"):
        analyse_alpha_sweep_records(records, protocol=_protocol())


def test_wrong_experiment_seed_is_rejected():
    records = list(_records())
    records[0] = replace(records[0], experiment_seed=999)
    with pytest.raises(ValueError, match="experiment seed"):
        analyse_alpha_sweep_records(records, protocol=_protocol())


def test_alpha_zero_fingerprint_mismatch_is_rejected():
    records = list(_records())
    target = next(
        i for i, record in enumerate(records)
        if record.replication_id == 0 and record.alpha == 0.0 and record.topology_label == "SF"
    )
    records[target] = replace(records[target], economic_path_fingerprint="different".ljust(64, "0"))
    with pytest.raises(ValueError, match="alpha=0 economic-path topology-null failed"):
        analyse_alpha_sweep_records(records, protocol=_protocol())
