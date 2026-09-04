from dataclasses import replace

import pytest

from src.experiments.refined.confirmatory_inference import analyse_confirmatory_records
from src.experiments.refined.confirmatory_protocol import ConfirmatoryProductionProtocol
from src.experiments.refined.confirmatory_runner import ConfirmatoryTreatmentRecord


def _protocol() -> ConfirmatoryProductionProtocol:
    return ConfirmatoryProductionProtocol(
        experiment_seed=92001,
        n_replications=4,
        bootstrap_seed=92002,
        n_bootstrap=1000,
    )


def _record(replication_id: int, topology: str) -> ConfirmatoryTreatmentRecord:
    topology_level = {"R": 1.0, "SW": 2.0, "SF": 3.0}[topology]
    common = 0.1 * replication_id
    level = topology_level + common
    threshold = topology == "SF" or (topology == "SW" and replication_id % 2 == 0)
    censored = topology == "SF" and replication_id == 3
    return ConfirmatoryTreatmentRecord(
        experiment_seed=92001,
        replication_id=replication_id,
        regime="baseline",
        alpha=0.75,
        topology_label=topology,
        graph_seed=1000 + 10 * replication_id + {"R": 1, "SW": 2, "SF": 3}[topology],
        shock_seed=2000 + replication_id,
        initial_state_seed=3000 + replication_id,
        economic_path_fingerprint=(topology + str(replication_id)).ljust(64, "0"),
        return_volatility=level,
        rms_mispricing=2.0 * level,
        maximum_absolute_mispricing=3.0 * level,
        mean_absolute_order_flow_per_agent=0.5 * level,
        mean_absolute_return=0.25 * level,
        time_averaged_belief_variance=0.1 * level,
        peak_cid=1.5 * level,
        threshold_exceeding=threshold,
        cid_exceedance_duration_share=0.05 * topology_level,
        stabilised=not censored,
        stabilisation_period=None if censored else 50 + replication_id,
        right_censored=censored,
        mean_pairwise_action_covariance={"R": -0.2, "SW": 0.0, "SF": 0.2}[topology] + 0.01 * replication_id,
        mean_sum_individual_action_variances=4.0 * level,
        mean_aggregate_order_flow_variance=5.0 * level,
        in_degree_gini=0.1 * topology_level,
        hub_link_share=0.1 * topology_level,
        global_clustering=0.1,
        average_path_length_lcc=2.0,
        largest_component_share=1.0,
        mean_attention_entropy=0.9 - 0.1 * topology_level,
        mean_effective_sources=7.0 - topology_level,
        mean_influence_hhi=0.02 * topology_level,
        mean_hub_influence_share=0.1 * topology_level,
        mean_attention_overlap=0.03 * topology_level,
        mean_attention_mobility=0.01 * topology_level,
    )


def _records():
    return tuple(
        _record(replication_id, topology)
        for replication_id in range(4)
        for topology in ("R", "SW", "SF")
    )


def _contrast(result, outcome: str, left: str, right: str):
    return next(
        item
        for item in result.pairwise_contrasts
        if item.outcome == outcome and item.topology_left == left and item.topology_right == right
    )


def test_point_means_and_pairwise_sign_follow_declared_subtraction_order():
    result = analyse_confirmatory_records(_records(), protocol=_protocol())
    means = {
        (item.outcome, item.topology): item.estimate
        for item in result.topology_means
    }
    assert means[("return_volatility", "R")] == pytest.approx(1.15)
    assert means[("return_volatility", "SF")] == pytest.approx(3.15)
    contrast = _contrast(result, "return_volatility", "R", "SF")
    assert contrast.estimate == pytest.approx(-2.0)
    assert contrast.relative_effect is not None


def test_binary_outcome_is_reported_as_absolute_probability_difference_only():
    result = analyse_confirmatory_records(_records(), protocol=_protocol())
    contrast = _contrast(result, "threshold_exceeding", "R", "SF")
    assert contrast.estimate == pytest.approx(-1.0)
    assert contrast.relative_effect is None
    assert contrast.relative_ci_lower is None
    assert contrast.relative_ci_upper is None


def test_signed_action_covariance_does_not_use_relative_effect():
    result = analyse_confirmatory_records(_records(), protocol=_protocol())
    contrast = _contrast(result, "mean_pairwise_action_covariance", "R", "SF")
    assert contrast.estimate == pytest.approx(-0.4)
    assert contrast.relative_effect is None


def test_primary_and_mechanism_families_receive_holm_adjustment():
    result = analyse_confirmatory_records(_records(), protocol=_protocol())
    primary = _contrast(result, "return_volatility", "R", "SF")
    mechanism = _contrast(result, "mean_attention_overlap", "R", "SF")
    secondary = _contrast(result, "mean_absolute_return", "R", "SF")
    assert primary.multiplicity_method == "holm_fwer"
    assert primary.adjusted_p_value is not None
    assert primary.reject_familywise is not None
    assert mechanism.multiplicity_method == "holm_fwer"
    assert mechanism.adjusted_p_value is not None
    assert secondary.multiplicity_method == "pointwise_exploratory"
    assert secondary.adjusted_p_value is None
    assert secondary.reject_familywise is None


def test_bootstrap_is_exactly_reproducible_for_fixed_bootstrap_seed():
    first = analyse_confirmatory_records(_records(), protocol=_protocol())
    second = analyse_confirmatory_records(_records(), protocol=_protocol())
    assert first == second


def test_censored_replications_are_counted_not_dropped():
    result = analyse_confirmatory_records(_records(), protocol=_protocol())
    assert dict(result.censored_counts) == {"R": 0, "SW": 0, "SF": 1}
    assert result.n_replications == 4


def test_incomplete_triplet_is_rejected():
    records = _records()[:-1]
    with pytest.raises(ValueError, match="complete matched"):
        analyse_confirmatory_records(records, protocol=_protocol())


def test_wrong_experiment_seed_is_rejected():
    records = list(_records())
    records[0] = replace(records[0], experiment_seed=99999)
    with pytest.raises(ValueError, match="experiment seed"):
        analyse_confirmatory_records(records, protocol=_protocol())


def test_full_sample_guard_requires_all_predeclared_replication_ids():
    protocol = replace(_protocol(), n_replications=5)
    with pytest.raises(ValueError, match="every predeclared replication"):
        analyse_confirmatory_records(_records(), protocol=protocol, require_full_sample=True)


def test_partial_analysis_is_allowed_only_when_explicitly_requested():
    protocol = replace(_protocol(), n_replications=5)
    result = analyse_confirmatory_records(
        _records(),
        protocol=protocol,
        require_full_sample=False,
    )
    assert result.n_replications == 4
