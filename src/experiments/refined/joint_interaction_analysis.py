"""Matched complete-block analysis for D049 joint alpha-beta-gamma_R interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .confirmatory_protocol import first_confirmatory_production_protocol
from .confirmatory_runner import ConfirmatoryTreatmentRecord
from .gamma_sweep_protocol import GAMMA_DIAGNOSTIC_OUTCOMES
from .joint_interaction_protocol import (
    JointInteractionProtocol,
    JointLinearContrastSpec,
)


@dataclass(frozen=True, slots=True)
class JointTreatmentRecord:
    cell_id: str
    cell_kind: str
    alpha: float
    beta: float
    gamma_R: float
    mean_raw_local_reputation_std: float
    mean_raw_local_reputation_std_over_sigma0: float
    treatment: ConfirmatoryTreatmentRecord

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id must be non-empty")
        if self.cell_kind not in {"factorial", "alpha0_control", "d043_anchor"}:
            raise ValueError("invalid D049 cell_kind")
        alpha = float(self.alpha)
        beta = float(self.beta)
        gamma_R = float(self.gamma_R)
        if not np.isfinite(alpha) or not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if not np.isfinite(gamma_R) or not 0.0 <= gamma_R < 1.0:
            raise ValueError("gamma_R must satisfy 0 <= gamma_R < 1")
        for name in (
            "mean_raw_local_reputation_std",
            "mean_raw_local_reputation_std_over_sigma0",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.treatment, ConfirmatoryTreatmentRecord):
            raise TypeError("treatment must be ConfirmatoryTreatmentRecord")
        if float(self.treatment.alpha) != alpha:
            raise ValueError("wrapped treatment alpha must equal the D049 cell alpha")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "gamma_R", gamma_R)


@dataclass(frozen=True, slots=True)
class JointTopologyMeanResult:
    family: str
    cell_id: str
    cell_kind: str
    alpha: float
    beta: float
    gamma_R: float
    outcome: str
    topology: str
    estimate: float
    ci_lower: float
    ci_upper: float
    n_replications: int


@dataclass(frozen=True, slots=True)
class JointTopologyGapResult:
    family: str
    cell_id: str
    cell_kind: str
    alpha: float
    beta: float
    gamma_R: float
    outcome: str
    absolute_gap: float
    absolute_gap_ci_lower: float
    absolute_gap_ci_upper: float
    relative_gap: float | None
    relative_gap_ci_lower: float | None
    relative_gap_ci_upper: float | None
    n_replications: int


@dataclass(frozen=True, slots=True)
class JointPairwiseContrastResult:
    family: str
    cell_id: str
    cell_kind: str
    alpha: float
    beta: float
    gamma_R: float
    outcome: str
    topology_left: str
    topology_right: str
    estimate: float
    ci_lower: float
    ci_upper: float
    relative_effect: float | None
    relative_ci_lower: float | None
    relative_ci_upper: float | None
    n_replications: int


@dataclass(frozen=True, slots=True)
class JointInteractionResult:
    family: str
    priority: str
    contrast_name: str
    contrast_kind: str
    outcome: str
    topology_left: str
    topology_right: str
    estimate: float
    ci_lower: float
    ci_upper: float
    n_replications: int


@dataclass(frozen=True, slots=True)
class JointInteractionAnalysisResult:
    experiment_seed: int
    bootstrap_seed: int
    n_replications: int
    n_bootstrap: int
    confidence_level: float
    n_cells: int
    cross_cell_common_random_numbers_verified: bool
    alpha_zero_economic_path_null_verified: bool
    topology_means: tuple[JointTopologyMeanResult, ...]
    topology_gaps: tuple[JointTopologyGapResult, ...]
    pairwise_contrasts: tuple[JointPairwiseContrastResult, ...]
    interactions: tuple[JointInteractionResult, ...]


def _family_for(outcome: str) -> str:
    if outcome in GAMMA_DIAGNOSTIC_OUTCOMES:
        return "mechanism"
    d045 = first_confirmatory_production_protocol()
    if outcome in d045.primary_outcomes:
        return "primary"
    if outcome in d045.mechanism_outcomes:
        return "mechanism"
    if outcome in d045.secondary_outcomes:
        return "secondary"
    raise KeyError(outcome)


def _numeric_value(record: JointTreatmentRecord, outcome: str) -> float:
    if outcome in GAMMA_DIAGNOSTIC_OUTCOMES:
        value = float(getattr(record, outcome))
    else:
        value = getattr(record.treatment, outcome)
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite D049 outcome {outcome!r}")
    return value


def _matched_tensor(
    records: Iterable[JointTreatmentRecord],
    protocol: JointInteractionProtocol,
    *,
    require_full_sample: bool,
) -> tuple[tuple[int, ...], np.ndarray, dict[int, dict[str, dict[str, JointTreatmentRecord]]]]:
    records = tuple(records)
    if not records:
        raise ValueError("records must contain D049 treatments")
    if not all(isinstance(record, JointTreatmentRecord) for record in records):
        raise TypeError("records must contain only JointTreatmentRecord objects")

    cell_by_id = {cell.cell_id: cell for cell in protocol.cells}
    by_replication: dict[int, dict[str, dict[str, JointTreatmentRecord]]] = {}
    for wrapped in records:
        treatment = wrapped.treatment
        if treatment.regime != "joint_interaction":
            raise ValueError("D049 analysis accepts joint_interaction records only")
        if treatment.experiment_seed != protocol.experiment_seed:
            raise ValueError("record experiment seed does not match D049")
        if wrapped.cell_id not in cell_by_id:
            raise ValueError(f"unknown D049 cell_id={wrapped.cell_id}")
        cell = cell_by_id[wrapped.cell_id]
        if (
            wrapped.cell_kind != cell.cell_kind
            or wrapped.alpha != cell.alpha
            or wrapped.beta != cell.beta
            or wrapped.gamma_R != cell.gamma_R
        ):
            raise ValueError("wrapped D049 cell metadata does not match the frozen protocol")
        bucket = by_replication.setdefault(treatment.replication_id, {}).setdefault(
            wrapped.cell_id, {}
        )
        if treatment.topology_label in bucket:
            raise ValueError("duplicate topology record in D049 cell/replication block")
        bucket[treatment.topology_label] = wrapped

    replication_ids = tuple(sorted(by_replication))
    if require_full_sample and replication_ids != tuple(range(protocol.n_replications)):
        raise ValueError("final D049 analysis requires every predeclared replication")
    if len(replication_ids) < 2:
        raise ValueError("D049 analysis requires at least two replications")

    expected_cells = {cell.cell_id for cell in protocol.cells}
    expected_topologies = set(protocol.topology_labels)
    alpha0_id = next(cell.cell_id for cell in protocol.cells if cell.cell_kind == "alpha0_control")
    alpha_zero_null = True

    for replication_id in replication_ids:
        cell_buckets = by_replication[replication_id]
        if set(cell_buckets) != expected_cells:
            raise ValueError(f"replication {replication_id} does not contain all D049 cells")
        reference_cell_id = protocol.cells[0].cell_id
        reference = cell_buckets[reference_cell_id]
        reference_shock = reference[protocol.topology_labels[0]].treatment.shock_seed
        reference_initial = reference[protocol.topology_labels[0]].treatment.initial_state_seed
        for cell in protocol.cells:
            bucket = cell_buckets[cell.cell_id]
            if set(bucket) != expected_topologies:
                raise ValueError(
                    f"replication {replication_id}, cell {cell.cell_id} is not a complete R/SW/SF triplet"
                )
            for topology in protocol.topology_labels:
                current = bucket[topology].treatment
                if current.shock_seed != reference_shock:
                    raise ValueError(f"cross-cell shock-seed mismatch in replication {replication_id}")
                if current.initial_state_seed != reference_initial:
                    raise ValueError(
                        f"cross-cell initial-state-seed mismatch in replication {replication_id}"
                    )
                if current.graph_seed != reference[topology].treatment.graph_seed:
                    raise ValueError(
                        f"cross-cell graph-seed mismatch in replication {replication_id}, topology={topology}"
                    )
        alpha0_fingerprints = {
            cell_buckets[alpha0_id][topology].treatment.economic_path_fingerprint
            for topology in protocol.topology_labels
        }
        if len(alpha0_fingerprints) != 1:
            alpha_zero_null = False

    values = np.empty(
        (
            len(replication_ids),
            protocol.n_cells,
            len(protocol.topology_labels),
            len(protocol.outcomes),
        ),
        dtype=float,
    )
    for r_index, replication_id in enumerate(replication_ids):
        for c_index, cell in enumerate(protocol.cells):
            bucket = by_replication[replication_id][cell.cell_id]
            for t_index, topology in enumerate(protocol.topology_labels):
                wrapped = bucket[topology]
                for o_index, outcome in enumerate(protocol.outcomes):
                    values[r_index, c_index, t_index, o_index] = _numeric_value(wrapped, outcome)

    return replication_ids, values, by_replication


def _bootstrap_means(
    values: np.ndarray,
    protocol: JointInteractionProtocol,
    *,
    batch_size: int = 100,
) -> np.ndarray:
    """Return B x cell x topology x outcome complete-block bootstrap means."""

    n_replications = values.shape[0]
    flattened = values.reshape(n_replications, -1)
    rng = np.random.default_rng(protocol.bootstrap_seed)
    probabilities = np.full(n_replications, 1.0 / n_replications, dtype=float)
    result = np.empty((protocol.n_bootstrap, flattened.shape[1]), dtype=float)
    for start in range(0, protocol.n_bootstrap, batch_size):
        stop = min(start + batch_size, protocol.n_bootstrap)
        counts = rng.multinomial(n_replications, probabilities, size=stop - start)
        result[start:stop] = (counts @ flattened) / n_replications
    return result.reshape((protocol.n_bootstrap,) + values.shape[1:])


def _percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    tail = 1.0 - confidence_level
    return (
        float(np.quantile(values, tail / 2.0, method="linear")),
        float(np.quantile(values, 1.0 - tail / 2.0, method="linear")),
    )


def _interaction_value(
    spec: JointLinearContrastSpec,
    cell_effects: np.ndarray,
    protocol: JointInteractionProtocol,
) -> float:
    total = 0.0
    for alpha, beta, gamma_R, coefficient in spec.terms:
        cell = protocol.factorial_cell(alpha, beta, gamma_R)
        total += coefficient * float(cell_effects[protocol.cell_index(cell.cell_id)])
    return float(total)


def _interaction_bootstrap(
    spec: JointLinearContrastSpec,
    cell_effects_boot: np.ndarray,
    protocol: JointInteractionProtocol,
) -> np.ndarray:
    total = np.zeros(cell_effects_boot.shape[0], dtype=float)
    for alpha, beta, gamma_R, coefficient in spec.terms:
        cell = protocol.factorial_cell(alpha, beta, gamma_R)
        total += coefficient * cell_effects_boot[:, protocol.cell_index(cell.cell_id)]
    return total


def analyse_joint_interaction_records(
    records: Iterable[JointTreatmentRecord],
    *,
    protocol: JointInteractionProtocol,
    require_full_sample: bool = True,
) -> JointInteractionAnalysisResult:
    """Compute D049 cell surfaces and predeclared signed interaction contrasts."""

    if not isinstance(protocol, JointInteractionProtocol):
        raise TypeError("protocol must be JointInteractionProtocol")
    replication_ids, values, _ = _matched_tensor(
        records,
        protocol,
        require_full_sample=require_full_sample,
    )
    n_replications = len(replication_ids)
    point_means = values.mean(axis=0)
    bootstrap_means = _bootstrap_means(values, protocol)

    topology_index = {label: i for i, label in enumerate(protocol.topology_labels)}
    outcome_index = {outcome: i for i, outcome in enumerate(protocol.outcomes)}
    mean_results: list[JointTopologyMeanResult] = []
    gap_results: list[JointTopologyGapResult] = []
    pair_results: list[JointPairwiseContrastResult] = []
    interaction_results: list[JointInteractionResult] = []

    for c_index, cell in enumerate(protocol.cells):
        for outcome in protocol.outcomes:
            family = _family_for(outcome)
            o_index = outcome_index[outcome]
            outcome_point = point_means[c_index, :, o_index]
            outcome_boot = bootstrap_means[:, c_index, :, o_index]

            for topology in protocol.topology_labels:
                t_index = topology_index[topology]
                lower, upper = _percentile_interval(
                    outcome_boot[:, t_index], protocol.confidence_level
                )
                mean_results.append(
                    JointTopologyMeanResult(
                        family=family,
                        cell_id=cell.cell_id,
                        cell_kind=cell.cell_kind,
                        alpha=cell.alpha,
                        beta=cell.beta,
                        gamma_R=cell.gamma_R,
                        outcome=outcome,
                        topology=topology,
                        estimate=float(outcome_point[t_index]),
                        ci_lower=lower,
                        ci_upper=upper,
                        n_replications=n_replications,
                    )
                )

            absolute_gap = float(np.max(outcome_point) - np.min(outcome_point))
            boot_abs_gap = np.max(outcome_boot, axis=1) - np.min(outcome_boot, axis=1)
            gap_lower, gap_upper = _percentile_interval(boot_abs_gap, protocol.confidence_level)
            relative_gap = relative_lower = relative_upper = None
            if protocol.uses_relative_effect(outcome):
                denominator = float(np.mean(outcome_point) + protocol.relative_epsilon)
                relative_gap = absolute_gap / denominator
                boot_rel_gap = boot_abs_gap / (
                    np.mean(outcome_boot, axis=1) + protocol.relative_epsilon
                )
                relative_lower, relative_upper = _percentile_interval(
                    boot_rel_gap, protocol.confidence_level
                )
            gap_results.append(
                JointTopologyGapResult(
                    family=family,
                    cell_id=cell.cell_id,
                    cell_kind=cell.cell_kind,
                    alpha=cell.alpha,
                    beta=cell.beta,
                    gamma_R=cell.gamma_R,
                    outcome=outcome,
                    absolute_gap=absolute_gap,
                    absolute_gap_ci_lower=gap_lower,
                    absolute_gap_ci_upper=gap_upper,
                    relative_gap=relative_gap,
                    relative_gap_ci_lower=relative_lower,
                    relative_gap_ci_upper=relative_upper,
                    n_replications=n_replications,
                )
            )

            for left, right in protocol.topology_pairs:
                left_index = topology_index[left]
                right_index = topology_index[right]
                estimate = float(outcome_point[left_index] - outcome_point[right_index])
                boot_contrast = outcome_boot[:, left_index] - outcome_boot[:, right_index]
                lower, upper = _percentile_interval(boot_contrast, protocol.confidence_level)
                relative_effect = relative_ci_lower = relative_ci_upper = None
                if protocol.uses_relative_effect(outcome):
                    denominator = float(
                        0.5 * (outcome_point[left_index] + outcome_point[right_index])
                        + protocol.relative_epsilon
                    )
                    relative_effect = estimate / denominator
                    boot_denominator = (
                        0.5 * (outcome_boot[:, left_index] + outcome_boot[:, right_index])
                        + protocol.relative_epsilon
                    )
                    boot_relative = boot_contrast / boot_denominator
                    relative_ci_lower, relative_ci_upper = _percentile_interval(
                        boot_relative, protocol.confidence_level
                    )
                pair_results.append(
                    JointPairwiseContrastResult(
                        family=family,
                        cell_id=cell.cell_id,
                        cell_kind=cell.cell_kind,
                        alpha=cell.alpha,
                        beta=cell.beta,
                        gamma_R=cell.gamma_R,
                        outcome=outcome,
                        topology_left=left,
                        topology_right=right,
                        estimate=estimate,
                        ci_lower=lower,
                        ci_upper=upper,
                        relative_effect=relative_effect,
                        relative_ci_lower=relative_ci_lower,
                        relative_ci_upper=relative_ci_upper,
                        n_replications=n_replications,
                    )
                )

    sf_index = topology_index[protocol.interaction_topology_pair[0]]
    sw_index = topology_index[protocol.interaction_topology_pair[1]]
    for outcome in protocol.outcomes:
        family = _family_for(outcome)
        o_index = outcome_index[outcome]
        cell_effects = point_means[:, sf_index, o_index] - point_means[:, sw_index, o_index]
        cell_effects_boot = (
            bootstrap_means[:, :, sf_index, o_index]
            - bootstrap_means[:, :, sw_index, o_index]
        )
        for spec in protocol.interaction_specs:
            estimate = _interaction_value(spec, cell_effects, protocol)
            boot_values = _interaction_bootstrap(spec, cell_effects_boot, protocol)
            lower, upper = _percentile_interval(boot_values, protocol.confidence_level)
            interaction_results.append(
                JointInteractionResult(
                    family=family,
                    priority="core" if outcome in protocol.core_outcomes else "diagnostic",
                    contrast_name=spec.name,
                    contrast_kind=spec.contrast_kind,
                    outcome=outcome,
                    topology_left=protocol.interaction_topology_pair[0],
                    topology_right=protocol.interaction_topology_pair[1],
                    estimate=estimate,
                    ci_lower=lower,
                    ci_upper=upper,
                    n_replications=n_replications,
                )
            )

    alpha0_id = next(cell.cell_id for cell in protocol.cells if cell.cell_kind == "alpha0_control")
    alpha0_index = protocol.cell_index(alpha0_id)
    alpha0_fingerprints_by_rep: dict[int, set[str]] = {}
    for wrapped in records:
        if wrapped.cell_id == alpha0_id:
            alpha0_fingerprints_by_rep.setdefault(wrapped.treatment.replication_id, set()).add(
                wrapped.treatment.economic_path_fingerprint
            )
    alpha_zero_null = all(len(items) == 1 for items in alpha0_fingerprints_by_rep.values())
    if len(alpha0_fingerprints_by_rep) != n_replications:
        alpha_zero_null = False
    _ = alpha0_index  # documents that the control is an explicit protocol cell

    return JointInteractionAnalysisResult(
        experiment_seed=protocol.experiment_seed,
        bootstrap_seed=protocol.bootstrap_seed,
        n_replications=n_replications,
        n_bootstrap=protocol.n_bootstrap,
        confidence_level=protocol.confidence_level,
        n_cells=protocol.n_cells,
        cross_cell_common_random_numbers_verified=True,
        alpha_zero_economic_path_null_verified=alpha_zero_null,
        topology_means=tuple(mean_results),
        topology_gaps=tuple(gap_results),
        pairwise_contrasts=tuple(pair_results),
        interactions=tuple(interaction_results),
    )
