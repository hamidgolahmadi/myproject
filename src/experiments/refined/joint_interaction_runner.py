"""D049 paired treatment runner for one frozen alpha-beta-gamma_R cell."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.model.refined import simulate_shock_path
from src.topologies.refined import diagnose_graph

from .action_covariance import rolling_action_covariance
from .baseline_specification import (
    RefinedBaselineSpecification,
    generate_neutral_nonnetwork_initial_conditions,
)
from .cid import rolling_cid
from .cid_events import CIDThresholdConfiguration, classify_cid_path
from .confirmatory_runner import ConfirmatoryTreatmentRecord, _economic_path_fingerprint
from .influence_metrics import realised_influence_path
from .joint_interaction_analysis import JointTreatmentRecord
from .joint_interaction_protocol import JointCell
from .market_calibration import MarketEvaluationCalibration
from .market_metrics import compute_run_level_market_outcomes
from .paired import prepare_paired_replication
from .reputation_diagnostics import reputation_dispersion_diagnostics
from .seeding import nonnegative_integer
from .treatments import prepare_paired_treatments


def run_paired_joint_replication(
    *,
    experiment_seed: int,
    replication_id: int,
    cell: JointCell,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> tuple[JointTreatmentRecord, ...]:
    """Run one matched R/SW/SF D049 replication for one frozen parameter cell."""

    experiment_seed = nonnegative_integer("experiment_seed", experiment_seed)
    replication_id = nonnegative_integer("replication_id", replication_id)
    if not isinstance(cell, JointCell):
        raise TypeError("cell must be JointCell")
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if not isinstance(calibration, MarketEvaluationCalibration):
        raise TypeError("calibration must be MarketEvaluationCalibration")
    if baseline.horizon != calibration.protocol.horizon:
        raise ValueError("baseline horizon must match calibration protocol horizon")

    parameters = replace(
        baseline.parameters,
        alpha=cell.alpha,
        beta=cell.beta,
        gamma_R=cell.gamma_R,
    )
    specifications = baseline.topology_specifications
    labels = tuple(spec.topology_label for spec in specifications)
    plan = prepare_paired_replication(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        topology_labels=labels,
        n_periods=baseline.horizon,
        n_agents=baseline.n_agents,
        parameters=parameters,
    )
    common_initial = generate_neutral_nonnetwork_initial_conditions(
        n_agents=baseline.n_agents,
        parameters=parameters,
        initial_state_seed=plan.seeds.initial_state_seed,
    )
    treatments = prepare_paired_treatments(
        plan=plan,
        specifications=specifications,
        initial_conditions=common_initial,
        parameters=parameters,
    )

    thresholds = CIDThresholdConfiguration(cid_threshold=calibration.cid_threshold)
    wrapped_records: list[JointTreatmentRecord] = []
    for treatment in treatments:
        simulation = simulate_shock_path(
            treatment.initial_state,
            treatment.shock_path,
            treatment.graph,
            treatment.parameters,
            adaptive_attention=True,
        )
        market = compute_run_level_market_outcomes(
            simulation,
            burn_in=calibration.protocol.burn_in,
        )
        cid_path = rolling_cid(
            simulation,
            window_length=calibration.protocol.rolling_window,
            burn_in=calibration.protocol.burn_in,
            scales=calibration.reference_scales,
            weights=calibration.cid_weights,
        )
        classification = classify_cid_path(
            cid_path,
            thresholds=thresholds,
            stabilisation_length=calibration.protocol.stabilisation_length,
        )
        action_covariance = rolling_action_covariance(
            simulation,
            window_length=calibration.protocol.rolling_window,
            burn_in=calibration.protocol.burn_in,
        )
        structural = diagnose_graph(treatment.graph, q=baseline.hub_q)
        influence = realised_influence_path(
            simulation,
            treatment.graph,
            q=baseline.hub_q,
        )
        influence_points = influence.points
        reputation = reputation_dispersion_diagnostics(
            simulation,
            treatment.graph,
            sigma_0=parameters.sigma_0,
        )

        treatment_record = ConfirmatoryTreatmentRecord(
            experiment_seed=experiment_seed,
            replication_id=replication_id,
            regime="joint_interaction",
            alpha=parameters.alpha,
            topology_label=treatment.topology_label,
            graph_seed=treatment.graph_seed,
            shock_seed=plan.seeds.shock_seed,
            initial_state_seed=plan.seeds.initial_state_seed,
            economic_path_fingerprint=_economic_path_fingerprint(simulation),
            return_volatility=market.return_volatility,
            rms_mispricing=market.rms_mispricing,
            maximum_absolute_mispricing=market.maximum_absolute_mispricing,
            mean_absolute_order_flow_per_agent=market.mean_absolute_order_flow_per_agent,
            mean_absolute_return=market.mean_absolute_return,
            time_averaged_belief_variance=market.time_averaged_belief_variance,
            peak_cid=classification.peak_cid,
            threshold_exceeding=classification.threshold_exceeding,
            cid_exceedance_duration_share=classification.cid_exceedance_duration_share,
            stabilised=classification.stabilisation.stabilised,
            stabilisation_period=classification.stabilisation.stabilisation_period,
            right_censored=classification.stabilisation.right_censored,
            mean_pairwise_action_covariance=float(
                np.mean([p.average_pairwise_action_covariance for p in action_covariance])
            ),
            mean_sum_individual_action_variances=float(
                np.mean([p.sum_individual_action_variances for p in action_covariance])
            ),
            mean_aggregate_order_flow_variance=float(
                np.mean([p.aggregate_order_flow_variance for p in action_covariance])
            ),
            in_degree_gini=structural.in_degree_gini,
            hub_link_share=structural.hub_link_share,
            global_clustering=structural.global_clustering,
            average_path_length_lcc=structural.average_path_length_lcc,
            largest_component_share=structural.largest_component_share,
            mean_attention_entropy=float(
                np.mean([point.mean_normalised_entropy for point in influence_points])
            ),
            mean_effective_sources=float(
                np.mean([point.mean_effective_sources for point in influence_points])
            ),
            mean_influence_hhi=float(
                np.mean([point.influence_hhi for point in influence_points])
            ),
            mean_hub_influence_share=float(
                np.mean([point.structural_hub_influence_share for point in influence_points])
            ),
            mean_attention_overlap=float(
                np.mean([point.attention_overlap for point in influence_points])
            ),
            mean_attention_mobility=float(
                np.mean([point.attention_mobility for point in influence_points])
            ),
        )
        wrapped_records.append(
            JointTreatmentRecord(
                cell_id=cell.cell_id,
                cell_kind=cell.cell_kind,
                alpha=cell.alpha,
                beta=cell.beta,
                gamma_R=cell.gamma_R,
                mean_raw_local_reputation_std=reputation.mean_raw_local_reputation_std,
                mean_raw_local_reputation_std_over_sigma0=(
                    reputation.mean_raw_local_reputation_std_over_sigma0
                ),
                treatment=treatment_record,
            )
        )

    return tuple(wrapped_records)
