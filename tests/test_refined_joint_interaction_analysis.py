from dataclasses import replace

import pytest

from src.experiments.refined.confirmatory_runner import ConfirmatoryTreatmentRecord
from src.experiments.refined.joint_interaction_analysis import (
    JointTreatmentRecord,
    analyse_joint_interaction_records,
)
from src.experiments.refined.joint_interaction_protocol import JointInteractionProtocol


def _protocol():
    return JointInteractionProtocol(
        experiment_seed=97301,
        bootstrap_seed=97302,
        n_replications=2,
        n_bootstrap=1000,
    )


def _wrapped(protocol, replication_id, cell, topology):
    base = 10.0 + 0.1 * replication_id
    d = cell.alpha * cell.beta * cell.gamma_R
    topo_add = {"SW": 0.0, "R": 0.5 * d, "SF": d}[topology]
    level = base + topo_add
    graph_seed = 1000 + 10 * replication_id + {"R": 1, "SW": 2, "SF": 3}[topology]
    if cell.cell_kind == "alpha0_control":
        fingerprint = f"alpha0-{replication_id}".ljust(64, "0")
    else:
        fingerprint = f"{cell.cell_id}-{topology}-{replication_id}".ljust(64, "0")
    treatment = ConfirmatoryTreatmentRecord(
        experiment_seed=protocol.experiment_seed,
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
        cid_exceedance_duration_share=0.01,
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


def _records(protocol=None):
    protocol = protocol or _protocol()
    return tuple(
        _wrapped(protocol, replication_id, cell, topology)
        for replication_id in range(protocol.n_replications)
        for cell in protocol.cells
        for topology in protocol.topology_labels
    )


def test_joint_analysis_validates_full_crn_and_alpha_zero_null():
    p = _protocol()
    result = analyse_joint_interaction_records(_records(p), protocol=p)
    assert result.cross_cell_common_random_numbers_verified is True
    assert result.alpha_zero_economic_path_null_verified is True
    assert result.n_cells == 50
    assert result.n_replications == 2


def test_joint_three_way_interaction_recovers_known_synthetic_effect():
    p = _protocol()
    result = analyse_joint_interaction_records(_records(p), protocol=p)
    item = next(
        x for x in result.interactions
        if x.contrast_name == "alpha_beta_gamma_core" and x.outcome == "return_volatility"
    )
    expected = (0.85 - 0.40) * (5.0 - 0.0) * (0.99 - 0.0)
    assert item.estimate == pytest.approx(expected)
    assert item.topology_left == "SF" and item.topology_right == "SW"
    assert item.priority == "core"


def test_joint_beta_gamma_core_interaction_recovers_known_effect():
    p = _protocol()
    result = analyse_joint_interaction_records(_records(p), protocol=p)
    item = next(
        x for x in result.interactions
        if x.contrast_name == "beta_gamma_core_at_alpha085"
        and x.outcome == "return_volatility"
    )
    assert item.estimate == pytest.approx(0.85 * 5.0 * 0.99)


def test_joint_gamma_boundary_shift_recovers_known_effect():
    p = _protocol()
    result = analyse_joint_interaction_records(_records(p), protocol=p)
    item = next(
        x for x in result.interactions
        if x.contrast_name == "gamma_boundary_shift_at_alpha085_beta5"
        and x.outcome == "return_volatility"
    )
    assert item.estimate == pytest.approx(0.85 * 5.0 * (0.999 - 0.99))
    assert item.contrast_kind == "boundary_shift"


def test_joint_cellwise_pairwise_contrast_signs_are_preserved():
    p = _protocol()
    result = analyse_joint_interaction_records(_records(p), protocol=p)
    cell = p.factorial_cell(0.85, 5.0, 0.99)
    contrasts = {
        (x.topology_left, x.topology_right): x.estimate
        for x in result.pairwise_contrasts
        if x.cell_id == cell.cell_id and x.outcome == "return_volatility"
    }
    d = 0.85 * 5.0 * 0.99
    assert contrasts[("R", "SW")] == pytest.approx(0.5 * d)
    assert contrasts[("R", "SF")] == pytest.approx(-0.5 * d)
    assert contrasts[("SW", "SF")] == pytest.approx(-d)


def test_joint_binary_and_signed_outcomes_have_no_relative_pairwise_effect():
    p = _protocol()
    result = analyse_joint_interaction_records(_records(p), protocol=p)
    binary = next(x for x in result.pairwise_contrasts if x.outcome == "threshold_exceeding")
    signed = next(
        x for x in result.pairwise_contrasts if x.outcome == "mean_pairwise_action_covariance"
    )
    assert binary.relative_effect is None
    assert signed.relative_effect is None


def test_joint_cross_cell_seed_mismatch_is_rejected():
    p = _protocol()
    records = list(_records(p))
    target = next(
        i for i, x in enumerate(records)
        if x.treatment.replication_id == 0 and x.cell_id == "F01" and x.treatment.topology_label == "SF"
    )
    records[target] = replace(
        records[target],
        treatment=replace(records[target].treatment, shock_seed=99999),
    )
    with pytest.raises(ValueError, match="cross-cell shock-seed mismatch"):
        analyse_joint_interaction_records(records, protocol=p)


def test_joint_missing_cell_triplet_is_rejected():
    p = _protocol()
    records = list(_records(p))
    records = [
        x for x in records
        if not (x.treatment.replication_id == 1 and x.cell_id == "F47")
    ]
    with pytest.raises(ValueError, match="all D049 cells"):
        analyse_joint_interaction_records(records, protocol=p)
