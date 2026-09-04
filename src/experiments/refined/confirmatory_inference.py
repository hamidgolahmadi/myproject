"""D045 inference for paired R/SW/SF confirmatory replications.

The bootstrap preserves the complete matched topology triplet within every
resampled replication, as required by Section 5.5.1 of the doctoral report.
Primary market/CID contrasts and mechanism contrasts form two separately
predeclared families with Holm family-wise error control. Secondary outcomes
receive pointwise bootstrap intervals and are explicitly exploratory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .confirmatory_protocol import ConfirmatoryProductionProtocol
from .confirmatory_runner import ConfirmatoryTreatmentRecord


@dataclass(frozen=True, slots=True)
class TopologyMeanResult:
    family: str
    outcome: str
    topology: str
    estimate: float
    ci_lower: float
    ci_upper: float
    n_replications: int


@dataclass(frozen=True, slots=True)
class TopologyGapResult:
    family: str
    outcome: str
    absolute_gap: float
    absolute_gap_ci_lower: float
    absolute_gap_ci_upper: float
    relative_gap: float | None
    relative_gap_ci_lower: float | None
    relative_gap_ci_upper: float | None
    n_replications: int


@dataclass(frozen=True, slots=True)
class PairwiseContrastResult:
    family: str
    outcome: str
    topology_left: str
    topology_right: str
    estimate: float
    ci_lower: float
    ci_upper: float
    relative_effect: float | None
    relative_ci_lower: float | None
    relative_ci_upper: float | None
    bootstrap_p_value: float
    adjusted_p_value: float | None
    multiplicity_method: str
    reject_familywise: bool | None
    n_replications: int


@dataclass(frozen=True, slots=True)
class ConfirmatoryInferenceResult:
    experiment_seed: int
    bootstrap_seed: int
    n_replications: int
    n_bootstrap: int
    confidence_level: float
    topology_means: tuple[TopologyMeanResult, ...]
    topology_gaps: tuple[TopologyGapResult, ...]
    pairwise_contrasts: tuple[PairwiseContrastResult, ...]
    censored_counts: tuple[tuple[str, int], ...]


def _family_for(outcome: str, protocol: ConfirmatoryProductionProtocol) -> str:
    if outcome in protocol.primary_outcomes:
        return "primary"
    if outcome in protocol.mechanism_outcomes:
        return "mechanism"
    if outcome in protocol.secondary_outcomes:
        return "secondary"
    raise KeyError(outcome)


def _numeric_value(record: ConfirmatoryTreatmentRecord, outcome: str) -> float:
    value = getattr(record, outcome)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite outcome {outcome!r}")
    return value


def _paired_matrix(
    records: Iterable[ConfirmatoryTreatmentRecord],
    protocol: ConfirmatoryProductionProtocol,
    *,
    require_full_sample: bool,
) -> tuple[tuple[int, ...], np.ndarray]:
    records = tuple(records)
    if len(records) == 0:
        raise ValueError("records must contain at least one treatment record")
    if not all(isinstance(record, ConfirmatoryTreatmentRecord) for record in records):
        raise TypeError("records must contain only ConfirmatoryTreatmentRecord objects")

    by_replication: dict[int, dict[str, ConfirmatoryTreatmentRecord]] = {}
    for record in records:
        if record.regime != "baseline":
            raise ValueError("D045 production inference accepts baseline-regime records only")
        if record.experiment_seed != protocol.experiment_seed:
            raise ValueError("record experiment seed does not match D045 production seed")
        bucket = by_replication.setdefault(record.replication_id, {})
        if record.topology_label in bucket:
            raise ValueError("duplicate topology record within a paired replication")
        bucket[record.topology_label] = record

    replication_ids = tuple(sorted(by_replication))
    expected_labels = set(protocol.topology_labels)
    for replication_id in replication_ids:
        labels = set(by_replication[replication_id])
        if labels != expected_labels:
            raise ValueError(
                f"replication {replication_id} is not a complete matched R/SW/SF triplet"
            )

    if require_full_sample:
        expected_ids = tuple(range(protocol.n_replications))
        if replication_ids != expected_ids:
            raise ValueError("final D045 inference requires every predeclared replication")

    n_replications = len(replication_ids)
    if n_replications < 2:
        raise ValueError("paired inference requires at least two complete replications")

    values = np.empty(
        (n_replications, len(protocol.topology_labels), len(protocol.all_outcomes)),
        dtype=float,
    )
    for r_index, replication_id in enumerate(replication_ids):
        for t_index, topology in enumerate(protocol.topology_labels):
            record = by_replication[replication_id][topology]
            for o_index, outcome in enumerate(protocol.all_outcomes):
                values[r_index, t_index, o_index] = _numeric_value(record, outcome)

    return replication_ids, values


def _bootstrap_topology_means(
    values: np.ndarray,
    protocol: ConfirmatoryProductionProtocol,
    *,
    batch_size: int = 250,
) -> np.ndarray:
    """Return B x topology x outcome triplet-preserving bootstrap means."""

    n_replications, n_topologies, n_outcomes = values.shape
    flattened = values.reshape(n_replications, n_topologies * n_outcomes)
    rng = np.random.default_rng(protocol.bootstrap_seed)
    probabilities = np.full(n_replications, 1.0 / n_replications, dtype=float)
    result = np.empty(
        (protocol.n_bootstrap, n_topologies * n_outcomes),
        dtype=float,
    )

    for start in range(0, protocol.n_bootstrap, batch_size):
        stop = min(start + batch_size, protocol.n_bootstrap)
        counts = rng.multinomial(
            n_replications,
            probabilities,
            size=stop - start,
        )
        result[start:stop] = (counts @ flattened) / n_replications

    return result.reshape(protocol.n_bootstrap, n_topologies, n_outcomes)


def _percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    return (
        float(np.quantile(values, alpha / 2.0, method="linear")),
        float(np.quantile(values, 1.0 - alpha / 2.0, method="linear")),
    )


def _centered_bootstrap_p_value(draws: np.ndarray, estimate: float) -> float:
    centered = np.asarray(draws, dtype=float) - float(estimate)
    exceedances = int(np.sum(np.abs(centered) >= abs(float(estimate))))
    return float((exceedances + 1) / (centered.size + 1))


def _holm_adjust(p_values: list[float]) -> tuple[list[float], list[bool]]:
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted, [value <= 0.05 for value in adjusted]


def analyse_confirmatory_records(
    records: Iterable[ConfirmatoryTreatmentRecord],
    *,
    protocol: ConfirmatoryProductionProtocol,
    require_full_sample: bool = True,
) -> ConfirmatoryInferenceResult:
    """Compute D045 topology means, gaps, paired contrasts, and bootstrap uncertainty."""

    if not isinstance(protocol, ConfirmatoryProductionProtocol):
        raise TypeError("protocol must be ConfirmatoryProductionProtocol")

    replication_ids, values = _paired_matrix(
        records,
        protocol,
        require_full_sample=require_full_sample,
    )
    n_replications = len(replication_ids)
    point_means = values.mean(axis=0)
    bootstrap_means = _bootstrap_topology_means(values, protocol)

    topology_index = {label: index for index, label in enumerate(protocol.topology_labels)}
    outcome_index = {outcome: index for index, outcome in enumerate(protocol.all_outcomes)}

    mean_results: list[TopologyMeanResult] = []
    gap_results: list[TopologyGapResult] = []
    contrast_results: list[PairwiseContrastResult] = []
    family_positions: dict[str, list[int]] = {"primary": [], "mechanism": []}

    for outcome in protocol.all_outcomes:
        family = _family_for(outcome, protocol)
        o_index = outcome_index[outcome]
        outcome_point_means = point_means[:, o_index]
        outcome_boot_means = bootstrap_means[:, :, o_index]

        for topology in protocol.topology_labels:
            t_index = topology_index[topology]
            lower, upper = _percentile_interval(
                outcome_boot_means[:, t_index],
                protocol.confidence_level,
            )
            mean_results.append(
                TopologyMeanResult(
                    family=family,
                    outcome=outcome,
                    topology=topology,
                    estimate=float(outcome_point_means[t_index]),
                    ci_lower=lower,
                    ci_upper=upper,
                    n_replications=n_replications,
                )
            )

        absolute_gap = float(np.max(outcome_point_means) - np.min(outcome_point_means))
        bootstrap_absolute_gap = (
            np.max(outcome_boot_means, axis=1) - np.min(outcome_boot_means, axis=1)
        )
        abs_lower, abs_upper = _percentile_interval(
            bootstrap_absolute_gap,
            protocol.confidence_level,
        )

        relative_gap: float | None = None
        relative_gap_ci_lower: float | None = None
        relative_gap_ci_upper: float | None = None
        if protocol.uses_relative_effect(outcome):
            denominator = float(np.mean(outcome_point_means) + protocol.relative_epsilon)
            relative_gap = absolute_gap / denominator
            bootstrap_relative_gap = bootstrap_absolute_gap / (
                np.mean(outcome_boot_means, axis=1) + protocol.relative_epsilon
            )
            relative_gap_ci_lower, relative_gap_ci_upper = _percentile_interval(
                bootstrap_relative_gap,
                protocol.confidence_level,
            )

        gap_results.append(
            TopologyGapResult(
                family=family,
                outcome=outcome,
                absolute_gap=absolute_gap,
                absolute_gap_ci_lower=abs_lower,
                absolute_gap_ci_upper=abs_upper,
                relative_gap=relative_gap,
                relative_gap_ci_lower=relative_gap_ci_lower,
                relative_gap_ci_upper=relative_gap_ci_upper,
                n_replications=n_replications,
            )
        )

        for left, right in protocol.topology_pairs:
            left_index = topology_index[left]
            right_index = topology_index[right]
            estimate = float(
                outcome_point_means[left_index] - outcome_point_means[right_index]
            )
            bootstrap_contrast = (
                outcome_boot_means[:, left_index] - outcome_boot_means[:, right_index]
            )
            lower, upper = _percentile_interval(
                bootstrap_contrast,
                protocol.confidence_level,
            )
            p_value = _centered_bootstrap_p_value(bootstrap_contrast, estimate)

            relative_effect: float | None = None
            relative_ci_lower: float | None = None
            relative_ci_upper: float | None = None
            if protocol.uses_relative_effect(outcome):
                denominator = float(
                    0.5
                    * (outcome_point_means[left_index] + outcome_point_means[right_index])
                    + protocol.relative_epsilon
                )
                relative_effect = estimate / denominator
                bootstrap_denominator = (
                    0.5
                    * (
                        outcome_boot_means[:, left_index]
                        + outcome_boot_means[:, right_index]
                    )
                    + protocol.relative_epsilon
                )
                bootstrap_relative = bootstrap_contrast / bootstrap_denominator
                relative_ci_lower, relative_ci_upper = _percentile_interval(
                    bootstrap_relative,
                    protocol.confidence_level,
                )

            index = len(contrast_results)
            if family in family_positions:
                family_positions[family].append(index)
            contrast_results.append(
                PairwiseContrastResult(
                    family=family,
                    outcome=outcome,
                    topology_left=left,
                    topology_right=right,
                    estimate=estimate,
                    ci_lower=lower,
                    ci_upper=upper,
                    relative_effect=relative_effect,
                    relative_ci_lower=relative_ci_lower,
                    relative_ci_upper=relative_ci_upper,
                    bootstrap_p_value=p_value,
                    adjusted_p_value=None,
                    multiplicity_method="holm_fwer" if family in family_positions else "pointwise_exploratory",
                    reject_familywise=None,
                    n_replications=n_replications,
                )
            )

    # Apply Holm separately to the predeclared primary and mechanism families.
    for family in ("primary", "mechanism"):
        positions = family_positions[family]
        raw_p = [contrast_results[index].bootstrap_p_value for index in positions]
        adjusted, rejected = _holm_adjust(raw_p)
        for index, adjusted_p, reject in zip(positions, adjusted, rejected, strict=True):
            old = contrast_results[index]
            contrast_results[index] = PairwiseContrastResult(
                family=old.family,
                outcome=old.outcome,
                topology_left=old.topology_left,
                topology_right=old.topology_right,
                estimate=old.estimate,
                ci_lower=old.ci_lower,
                ci_upper=old.ci_upper,
                relative_effect=old.relative_effect,
                relative_ci_lower=old.relative_ci_lower,
                relative_ci_upper=old.relative_ci_upper,
                bootstrap_p_value=old.bootstrap_p_value,
                adjusted_p_value=adjusted_p,
                multiplicity_method=old.multiplicity_method,
                reject_familywise=reject,
                n_replications=old.n_replications,
            )

    right_censored_index = outcome_index["right_censored"]
    censored_counts = tuple(
        (
            topology,
            int(np.sum(values[:, topology_index[topology], right_censored_index])),
        )
        for topology in protocol.topology_labels
    )

    return ConfirmatoryInferenceResult(
        experiment_seed=protocol.experiment_seed,
        bootstrap_seed=protocol.bootstrap_seed,
        n_replications=n_replications,
        n_bootstrap=protocol.n_bootstrap,
        confidence_level=protocol.confidence_level,
        topology_means=tuple(mean_results),
        topology_gaps=tuple(gap_results),
        pairwise_contrasts=tuple(contrast_results),
        censored_counts=censored_counts,
    )
