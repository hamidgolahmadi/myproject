"""Matched-block analysis for the exploratory D046 alpha sweep.

Bootstrap resampling preserves the complete replication block: all frozen alpha
values and all R/SW/SF topology treatments move together.  D046 is explicitly
exploratory, so this module reports percentile intervals without converting the
sweep into a second family-wise confirmatory exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .alpha_sweep_protocol import AlphaSweepProtocol
from .confirmatory_protocol import first_confirmatory_production_protocol
from .confirmatory_runner import ConfirmatoryTreatmentRecord


@dataclass(frozen=True, slots=True)
class AlphaTopologyMeanResult:
    family: str
    alpha: float
    outcome: str
    topology: str
    estimate: float
    ci_lower: float
    ci_upper: float
    n_replications: int


@dataclass(frozen=True, slots=True)
class AlphaTopologyGapResult:
    family: str
    alpha: float
    outcome: str
    absolute_gap: float
    absolute_gap_ci_lower: float
    absolute_gap_ci_upper: float
    relative_gap: float | None
    relative_gap_ci_lower: float | None
    relative_gap_ci_upper: float | None
    n_replications: int


@dataclass(frozen=True, slots=True)
class AlphaPairwiseContrastResult:
    family: str
    alpha: float
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
class AlphaSweepAnalysisResult:
    experiment_seed: int
    bootstrap_seed: int
    n_replications: int
    n_bootstrap: int
    confidence_level: float
    alpha_grid: tuple[float, ...]
    alpha_zero_economic_path_null_verified: bool
    topology_means: tuple[AlphaTopologyMeanResult, ...]
    topology_gaps: tuple[AlphaTopologyGapResult, ...]
    pairwise_contrasts: tuple[AlphaPairwiseContrastResult, ...]


def _family_for(outcome: str) -> str:
    d045 = first_confirmatory_production_protocol()
    if outcome in d045.primary_outcomes:
        return "primary"
    if outcome in d045.mechanism_outcomes:
        return "mechanism"
    if outcome in d045.secondary_outcomes:
        return "secondary"
    raise KeyError(outcome)


def _numeric_value(record: ConfirmatoryTreatmentRecord, outcome: str) -> float:
    value = getattr(record, outcome)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite D046 outcome {outcome!r}")
    return value


def _matched_tensor(
    records: Iterable[ConfirmatoryTreatmentRecord],
    protocol: AlphaSweepProtocol,
    *,
    require_full_sample: bool,
) -> tuple[tuple[int, ...], np.ndarray]:
    records = tuple(records)
    if len(records) == 0:
        raise ValueError("records must contain at least one treatment record")
    if not all(isinstance(record, ConfirmatoryTreatmentRecord) for record in records):
        raise TypeError("records must contain only ConfirmatoryTreatmentRecord objects")

    by_replication: dict[int, dict[float, dict[str, ConfirmatoryTreatmentRecord]]] = {}
    alpha_set = set(protocol.alpha_grid)
    for record in records:
        if record.regime != "alpha_sweep":
            raise ValueError("D046 analysis accepts alpha_sweep records only")
        if record.experiment_seed != protocol.experiment_seed:
            raise ValueError("record experiment seed does not match D046")
        alpha = float(record.alpha)
        if alpha not in alpha_set:
            raise ValueError(f"record alpha={alpha} is not in the frozen D046 grid")
        alpha_bucket = by_replication.setdefault(record.replication_id, {}).setdefault(alpha, {})
        if record.topology_label in alpha_bucket:
            raise ValueError("duplicate topology record within a D046 alpha/replication block")
        alpha_bucket[record.topology_label] = record

    replication_ids = tuple(sorted(by_replication))
    if require_full_sample and replication_ids != tuple(range(protocol.n_replications)):
        raise ValueError("final D046 analysis requires every predeclared replication")
    if len(replication_ids) < 2:
        raise ValueError("D046 analysis requires at least two replications")

    expected_topologies = set(protocol.topology_labels)
    for replication_id in replication_ids:
        alpha_buckets = by_replication[replication_id]
        if set(alpha_buckets) != alpha_set:
            raise ValueError(
                f"replication {replication_id} does not contain the complete D046 alpha grid"
            )
        for alpha in protocol.alpha_grid:
            if set(alpha_buckets[alpha]) != expected_topologies:
                raise ValueError(
                    f"replication {replication_id}, alpha={alpha} is not a complete R/SW/SF triplet"
                )

        zero_records = alpha_buckets[0.0]
        fingerprints = {
            zero_records[topology].economic_path_fingerprint
            for topology in protocol.topology_labels
        }
        if len(fingerprints) != 1:
            raise ValueError(
                f"alpha=0 economic-path topology-null failed in replication {replication_id}"
            )

    values = np.empty(
        (
            len(replication_ids),
            len(protocol.alpha_grid),
            len(protocol.topology_labels),
            len(protocol.outcomes),
        ),
        dtype=float,
    )
    for r_index, replication_id in enumerate(replication_ids):
        for a_index, alpha in enumerate(protocol.alpha_grid):
            bucket = by_replication[replication_id][alpha]
            for t_index, topology in enumerate(protocol.topology_labels):
                record = bucket[topology]
                for o_index, outcome in enumerate(protocol.outcomes):
                    values[r_index, a_index, t_index, o_index] = _numeric_value(record, outcome)

    return replication_ids, values


def _bootstrap_means(
    values: np.ndarray,
    protocol: AlphaSweepProtocol,
    *,
    batch_size: int = 250,
) -> np.ndarray:
    """Return B x alpha x topology x outcome matched-block bootstrap means."""

    n_replications = values.shape[0]
    flattened = values.reshape(n_replications, -1)
    rng = np.random.default_rng(protocol.bootstrap_seed)
    probabilities = np.full(n_replications, 1.0 / n_replications, dtype=float)
    result = np.empty((protocol.n_bootstrap, flattened.shape[1]), dtype=float)

    for start in range(0, protocol.n_bootstrap, batch_size):
        stop = min(start + batch_size, protocol.n_bootstrap)
        counts = rng.multinomial(
            n_replications,
            probabilities,
            size=stop - start,
        )
        result[start:stop] = (counts @ flattened) / n_replications

    return result.reshape((protocol.n_bootstrap,) + values.shape[1:])


def _percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = 1.0 - confidence_level
    return (
        float(np.quantile(values, alpha / 2.0, method="linear")),
        float(np.quantile(values, 1.0 - alpha / 2.0, method="linear")),
    )


def analyse_alpha_sweep_records(
    records: Iterable[ConfirmatoryTreatmentRecord],
    *,
    protocol: AlphaSweepProtocol,
    require_full_sample: bool = True,
) -> AlphaSweepAnalysisResult:
    """Compute exploratory D046 curves with complete-block bootstrap intervals."""

    if not isinstance(protocol, AlphaSweepProtocol):
        raise TypeError("protocol must be AlphaSweepProtocol")

    replication_ids, values = _matched_tensor(
        records,
        protocol,
        require_full_sample=require_full_sample,
    )
    n_replications = len(replication_ids)
    point_means = values.mean(axis=0)
    bootstrap_means = _bootstrap_means(values, protocol)

    topology_index = {label: i for i, label in enumerate(protocol.topology_labels)}
    outcome_index = {outcome: i for i, outcome in enumerate(protocol.outcomes)}

    mean_results: list[AlphaTopologyMeanResult] = []
    gap_results: list[AlphaTopologyGapResult] = []
    contrast_results: list[AlphaPairwiseContrastResult] = []

    for a_index, alpha in enumerate(protocol.alpha_grid):
        for outcome in protocol.outcomes:
            family = _family_for(outcome)
            o_index = outcome_index[outcome]
            outcome_point = point_means[a_index, :, o_index]
            outcome_boot = bootstrap_means[:, a_index, :, o_index]

            for topology in protocol.topology_labels:
                t_index = topology_index[topology]
                lower, upper = _percentile_interval(
                    outcome_boot[:, t_index], protocol.confidence_level
                )
                mean_results.append(
                    AlphaTopologyMeanResult(
                        family=family,
                        alpha=alpha,
                        outcome=outcome,
                        topology=topology,
                        estimate=float(outcome_point[t_index]),
                        ci_lower=lower,
                        ci_upper=upper,
                        n_replications=n_replications,
                    )
                )

            absolute_gap = float(np.max(outcome_point) - np.min(outcome_point))
            bootstrap_absolute_gap = (
                np.max(outcome_boot, axis=1) - np.min(outcome_boot, axis=1)
            )
            gap_lower, gap_upper = _percentile_interval(
                bootstrap_absolute_gap, protocol.confidence_level
            )

            relative_gap: float | None = None
            relative_gap_lower: float | None = None
            relative_gap_upper: float | None = None
            if protocol.uses_relative_effect(outcome):
                denominator = float(np.mean(outcome_point) + protocol.relative_epsilon)
                relative_gap = absolute_gap / denominator
                bootstrap_relative_gap = bootstrap_absolute_gap / (
                    np.mean(outcome_boot, axis=1) + protocol.relative_epsilon
                )
                relative_gap_lower, relative_gap_upper = _percentile_interval(
                    bootstrap_relative_gap, protocol.confidence_level
                )

            gap_results.append(
                AlphaTopologyGapResult(
                    family=family,
                    alpha=alpha,
                    outcome=outcome,
                    absolute_gap=absolute_gap,
                    absolute_gap_ci_lower=gap_lower,
                    absolute_gap_ci_upper=gap_upper,
                    relative_gap=relative_gap,
                    relative_gap_ci_lower=relative_gap_lower,
                    relative_gap_ci_upper=relative_gap_upper,
                    n_replications=n_replications,
                )
            )

            for left, right in protocol.topology_pairs:
                left_index = topology_index[left]
                right_index = topology_index[right]
                estimate = float(outcome_point[left_index] - outcome_point[right_index])
                bootstrap_contrast = (
                    outcome_boot[:, left_index] - outcome_boot[:, right_index]
                )
                lower, upper = _percentile_interval(
                    bootstrap_contrast, protocol.confidence_level
                )

                relative_effect: float | None = None
                relative_lower: float | None = None
                relative_upper: float | None = None
                if protocol.uses_relative_effect(outcome):
                    denominator = float(
                        0.5 * (outcome_point[left_index] + outcome_point[right_index])
                        + protocol.relative_epsilon
                    )
                    relative_effect = estimate / denominator
                    bootstrap_denominator = (
                        0.5
                        * (outcome_boot[:, left_index] + outcome_boot[:, right_index])
                        + protocol.relative_epsilon
                    )
                    bootstrap_relative = bootstrap_contrast / bootstrap_denominator
                    relative_lower, relative_upper = _percentile_interval(
                        bootstrap_relative, protocol.confidence_level
                    )

                contrast_results.append(
                    AlphaPairwiseContrastResult(
                        family=family,
                        alpha=alpha,
                        outcome=outcome,
                        topology_left=left,
                        topology_right=right,
                        estimate=estimate,
                        ci_lower=lower,
                        ci_upper=upper,
                        relative_effect=relative_effect,
                        relative_ci_lower=relative_lower,
                        relative_ci_upper=relative_upper,
                        n_replications=n_replications,
                    )
                )

    return AlphaSweepAnalysisResult(
        experiment_seed=protocol.experiment_seed,
        bootstrap_seed=protocol.bootstrap_seed,
        n_replications=n_replications,
        n_bootstrap=protocol.n_bootstrap,
        confidence_level=protocol.confidence_level,
        alpha_grid=protocol.alpha_grid,
        alpha_zero_economic_path_null_verified=True,
        topology_means=tuple(mean_results),
        topology_gaps=tuple(gap_results),
        pairwise_contrasts=tuple(contrast_results),
    )
