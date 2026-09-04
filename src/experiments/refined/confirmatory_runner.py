"""Paired confirmatory fixed-topology market runner.

This module executes the frozen first-stage R/SW/SF design under common random
numbers. It consumes D043 for the market specification and the frozen D042
numerical calibration for CID evaluation. It produces treatment-level market,
CID, structural, and realised-influence diagnostics without estimating or
ranking topology effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.model.refined import SimulationResult, simulate_shock_path
from src.topologies.refined import diagnose_graph

from .baseline_specification import (
    RefinedBaselineSpecification,
    first_refined_baseline_specification,
    generate_neutral_nonnetwork_initial_conditions,
)
from .cid import rolling_cid
from .cid_events import CIDThresholdConfiguration, classify_cid_path
from .frozen_market_calibration import first_frozen_market_evaluation_calibration
from .influence_metrics import realised_influence_path
from .market_calibration import MarketEvaluationCalibration
from .market_metrics import compute_run_level_market_outcomes
from .paired import prepare_paired_replication
from .seeding import nonnegative_integer
from .treatments import prepare_paired_treatments


@dataclass(frozen=True, slots=True)
class ConfirmatoryTreatmentRecord:
    experiment_seed: int
    replication_id: int
    regime: str
    alpha: float
    topology_label: str
    graph_seed: int
    shock_seed: int
    initial_state_seed: int
    economic_path_fingerprint: str
    return_volatility: float
    rms_mispricing: float
    maximum_absolute_mispricing: float
    mean_absolute_order_flow_per_agent: float
    mean_absolute_return: float
    time_averaged_belief_variance: float
    peak_cid: float
    threshold_exceeding: bool
    cid_exceedance_duration_share: float
    stabilised: bool
    stabilisation_period: int | None
    right_censored: bool
    in_degree_gini: float
    hub_link_share: float
    global_clustering: float
    average_path_length_lcc: float
    largest_component_share: float
    mean_attention_entropy: float
    mean_effective_sources: float
    mean_influence_hhi: float
    mean_hub_influence_share: float
    mean_attention_overlap: float
    mean_attention_mobility: float


def _economic_path_fingerprint(result: SimulationResult) -> str:
    """Hash the complete non-attention economic state/output path."""

    digest = hashlib.sha256()
    arrays = (
        np.asarray([state.theta for state in result.states], dtype=np.float64),
        np.asarray([state.price for state in result.states], dtype=np.float64),
        np.stack([state.beliefs for state in result.states]).astype(np.float64),
        np.stack([state.positions for state in result.states]).astype(np.float64),
        np.stack([state.reputation for state in result.states]).astype(np.float64),
        np.asarray([output.return_ for output in result.period_outputs], dtype=np.float64),
        np.asarray([output.net_order_flow for output in result.period_outputs], dtype=np.float64),
        np.stack([output.actions for output in result.period_outputs]).astype(np.float64),
    )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_inputs(
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> None:
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if not isinstance(calibration, MarketEvaluationCalibration):
        raise TypeError("calibration must be MarketEvaluationCalibration")
    if baseline.horizon != calibration.protocol.horizon:
        raise ValueError("baseline horizon must match calibration protocol horizon")


def run_paired_confirmatory_replication(
    *,
    experiment_seed: int,
    replication_id: int,
    regime: str = "baseline",
    alpha_override: float | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> tuple[ConfirmatoryTreatmentRecord, ...]:
    """Run one matched R/SW/SF replication and return treatment records."""

    experiment_seed = nonnegative_integer("experiment_seed", experiment_seed)
    replication_id = nonnegative_integer("replication_id", replication_id)
    if not isinstance(regime, str) or regime == "" or regime != regime.strip():
        raise ValueError("regime must be a non-empty string without surrounding whitespace")
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_inputs(baseline, calibration)

    parameters = baseline.parameters
    if alpha_override is not None:
        if isinstance(alpha_override, bool):
            raise TypeError("alpha_override must be a real scalar")
        alpha = float(alpha_override)
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha_override must lie in [0,1]")
        parameters = replace(parameters, alpha=alpha)

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
    records: list[ConfirmatoryTreatmentRecord] = []
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
        structural = diagnose_graph(treatment.graph, q=baseline.hub_q)
        influence = realised_influence_path(
            simulation,
            treatment.graph,
            q=baseline.hub_q,
        )
        influence_points = influence.points

        records.append(
            ConfirmatoryTreatmentRecord(
                experiment_seed=experiment_seed,
                replication_id=replication_id,
                regime=regime,
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
        )

    return tuple(records)


@dataclass(frozen=True, slots=True)
class ConfirmatorySmokeResult:
    experiment_seed: int
    n_replications: int
    baseline: RefinedBaselineSpecification
    calibration: MarketEvaluationCalibration
    records: tuple[ConfirmatoryTreatmentRecord, ...]

    def __post_init__(self) -> None:
        _validate_inputs(self.baseline, self.calibration)
        expected = self.n_replications * 2 * 3
        if len(self.records) != expected:
            raise ValueError("unexpected number of smoke treatment records")


def run_paired_confirmatory_smoke(
    *,
    experiment_seed: int = 2026090401,
    n_replications: int = 2,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> ConfirmatorySmokeResult:
    """Run a small baseline + alpha=0 paired smoke without topology ranking."""

    experiment_seed = nonnegative_integer("experiment_seed", experiment_seed)
    if isinstance(n_replications, bool) or not isinstance(n_replications, (int, np.integer)):
        raise TypeError("n_replications must be an integer")
    n_replications = int(n_replications)
    if n_replications < 1:
        raise ValueError("n_replications must be positive")
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_inputs(baseline, calibration)

    records: list[ConfirmatoryTreatmentRecord] = []
    for replication_id in range(n_replications):
        records.extend(
            run_paired_confirmatory_replication(
                experiment_seed=experiment_seed,
                replication_id=replication_id,
                regime="baseline",
                baseline=baseline,
                calibration=calibration,
            )
        )
        records.extend(
            run_paired_confirmatory_replication(
                experiment_seed=experiment_seed,
                replication_id=replication_id,
                regime="alpha0_control",
                alpha_override=0.0,
                baseline=baseline,
                calibration=calibration,
            )
        )

    return ConfirmatorySmokeResult(
        experiment_seed=experiment_seed,
        n_replications=n_replications,
        baseline=baseline,
        calibration=calibration,
        records=tuple(records),
    )


def write_paired_confirmatory_smoke(
    result: ConfirmatorySmokeResult,
    *,
    outdir: str | Path,
) -> tuple[Path, Path]:
    """Persist smoke records and metadata; no topology ranking is produced."""

    if not isinstance(result, ConfirmatorySmokeResult):
        raise TypeError("result must be ConfirmatorySmokeResult")
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "confirmatory_smoke_records.csv"
    metadata_path = output_dir / "confirmatory_smoke_metadata.json"

    rows = [asdict(record) for record in result.records]
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    calibration = result.calibration
    metadata = {
        "purpose": "paired R/SW/SF confirmatory pipeline smoke; not treatment-effect estimation",
        "final_confirmatory": False,
        "experiment_seed": result.experiment_seed,
        "n_replications": result.n_replications,
        "n_records": len(result.records),
        "topology_labels": [spec.topology_label for spec in result.baseline.topology_specifications],
        "regimes": ["baseline", "alpha0_control"],
        "baseline_alpha": result.baseline.parameters.alpha,
        "calibration": {
            "c_ret": calibration.reference_scales.return_scale,
            "c_bel": calibration.reference_scales.belief_scale,
            "c_F": calibration.reference_scales.order_flow_scale,
            "c_CID": calibration.cid_threshold,
            "rolling_window": calibration.protocol.rolling_window,
            "burn_in": calibration.protocol.burn_in,
            "stabilisation_length": calibration.protocol.stabilisation_length,
        },
        "interpretation_guard": "do not rank topologies from this smoke",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records_path, metadata_path
