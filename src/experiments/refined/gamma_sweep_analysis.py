"""Matched-block analysis for the exploratory D048 gamma_R sweep.

Bootstrap resampling preserves the complete replication block: every frozen
gamma_R value and all R/SW/SF topology treatments move together. D048 is an
OAT regime-mapping exercise, so percentile intervals are reported without
creating a new confirmatory multiple-testing family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .confirmatory_protocol import first_confirmatory_production_protocol
from .confirmatory_runner import ConfirmatoryTreatmentRecord
from .gamma_sweep_protocol import (
    GAMMA_DIAGNOSTIC_OUTCOMES,
    GammaSweepProtocol,
)


@dataclass(frozen=True, slots=True)
class GammaSweepTreatmentRecord:
    """One D048 treatment plus gamma-specific reputation-scale diagnostics."""

    gamma_R: float
    mean_raw_local_reputation_std: float
    mean_raw_local_reputation_std_over_sigma0: float
    treatment: ConfirmatoryTreatmentRecord

    def __post_init__(self) -> None:
        gamma_R = float(self.gamma_R)
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
        object.__setattr__(self, "gamma_R", gamma_R)


@dataclass(frozen=True, slots=True)
class GammaTopologyMeanResult:
    family: str
    gamma_R: float
    outcome: str
    topology: str
    estimate: float
    ci_lower: float
    ci_upper: float
    n_replications: int


@dataclass(frozen=True, slots=True)
class GammaTopologyGapResult:
    family: str
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
class GammaPairwiseContrastResult:
    family: str
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
class GammaSweepAnalysisResult:
    experiment_seed: int
    bootstrap_seed: int
    n_replications: int
    n_bootstrap: int
    confidence_level: float
    alpha_anchor: float
    beta_anchor: float
    gamma_grid: tuple[float, ...]
    cross_gamma_common_random_numbers_verified: bool
    topology_means: tuple[GammaTopologyMeanResult, ...]
    topology_gaps: tuple[GammaTopologyGapResult, ...]
    pairwise_contrasts: tuple[GammaPairwiseContrastResult, ...]


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


def _numeric_value(record: GammaSweepTreatmentRecord, outcome: str) -> float:
    if outcome in GAMMA_DIAGNOSTIC_OUTCOMES:
        value = float(getattr(record, outcome))
    else:
        value = getattr(record.treatment, outcome)
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite D048 outcome {outcome!r}")
    return value


def _matched_tensor(
    records: Iterable[GammaSweepTreatmentRecord],
    protocol: GammaSweepProtocol,
    *,
    require_full_sample: bool,
) -> tuple[tuple[int, ...], np.ndarray]:
    records = tuple(records)
    if len(records) == 0:
        raise ValueError("records must contain at least one D048 treatment record")
    if not all(isinstance(record, GammaSweepTreatmentRecord) for record in records):
        raise TypeError("records must contain only GammaSweepTreatmentRecord objects")

    by_replication: dict[int, dict[float, dict[str, GammaSweepTreatmentRecord]]] = {}
    gamma_set = set(protocol.gamma_grid)
    for wrapped in records:
        record = wrapped.treatment
        if record.regime != "gamma_sweep":
            raise ValueError("D048 analysis accepts gamma_sweep records only")
        if record.experiment_seed != protocol.experiment_seed:
            raise ValueError("record experiment seed does not match D048")
        if float(record.alpha) != protocol.alpha_anchor:
            raise ValueError("record alpha does not match the frozen D048 anchor")
        gamma_R = wrapped.gamma_R
        if gamma_R not in gamma_set:
            raise ValueError(f"record gamma_R={gamma_R} is not in the frozen D048 grid")
        bucket = by_replication.setdefault(record.replication_id, {}).setdefault(gamma_R, {})
        if record.topology_label in bucket:
            raise ValueError("duplicate topology record within a D048 gamma/replication block")
        bucket[record.topology_label] = wrapped

    replication_ids = tuple(sorted(by_replication))
    if require_full_sample and replication_ids != tuple(range(protocol.n_replications)):
        raise ValueError("final D048 analysis requires every predeclared replication")
    if len(replication_ids) < 2:
        raise ValueError("D048 analysis requires at least two replications")

    expected_topologies = set(protocol.topology_labels)
    for replication_id in replication_ids:
        gamma_buckets = by_replication[replication_id]
        if set(gamma_buckets) != gamma_set:
            raise ValueError(
                f"replication {replication_id} does not contain the complete D048 gamma grid"
            )
        for gamma_R in protocol.gamma_grid:
            if set(gamma_buckets[gamma_R]) != expected_topologies:
                raise ValueError(
                    f"replication {replication_id}, gamma_R={gamma_R} is not a complete R/SW/SF triplet"
                )

        reference_gamma = protocol.gamma_grid[0]
        reference = gamma_buckets[reference_gamma]
        reference_shock_seed = reference[protocol.topology_labels[0]].treatment.shock_seed
        reference_initial_seed = reference[protocol.topology_labels[0]].treatment.initial_state_seed
        for gamma_R in protocol.gamma_grid:
            bucket = gamma_buckets[gamma_R]
            for topology in protocol.topology_labels:
                current = bucket[topology].treatment
                if current.shock_seed != reference_shock_seed:
                    raise ValueError(
                        f"cross-gamma shock-seed mismatch in replication {replication_id}"
                    )
                if current.initial_state_seed != reference_initial_seed:
                    raise ValueError(
                        f"cross-gamma initial-state-seed mismatch in replication {replication_id}"
                    )
                if current.graph_seed != reference[topology].treatment.graph_seed:
                    raise ValueError(
                        f"cross-gamma graph-seed mismatch in replication {replication_id}, topology={topology}"
                    )

    values = np.empty(
        (
            len(replication_ids),
            len(protocol.gamma_grid),
            len(protocol.topology_labels),
            len(protocol.outcomes),
        ),
        dtype=float,
    )
    for r_index, replication_id in enumerate(replication_ids):
        for g_index, gamma_R in enumerate(protocol.gamma_grid):
            bucket = by_replication[replication_id][gamma_R]
            for t_index, topology in enumerate(protocol.topology_labels):
                wrapped = bucket[topology]
                for o_index, outcome in enumerate(protocol.outcomes):
                    values[r_index, g_index, t_index, o_index] = _numeric_value(wrapped, outcome)

    return replication_ids, values


def _bootstrap_means(
    values: np.ndarray,
    protocol: GammaSweepProtocol,
    *,
    batch_size: int = 250,
) -> np.ndarray:
    """Return B x gamma x topology x outcome matched-block bootstrap means."""

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


def analyse_gamma_sweep_records(
    records: Iterable[GammaSweepTreatmentRecord],
    *,
    protocol: GammaSweepProtocol,
    require_full_sample: bool = True,
) -> GammaSweepAnalysisResult:
    """Compute exploratory D048 curves with complete-block bootstrap intervals."""

    if not isinstance(protocol, GammaSweepProtocol):
        raise TypeError("protocol must be GammaSweepProtocol")
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
    mean_results: list[GammaTopologyMeanResult] = []
    gap_results: list[GammaTopologyGapResult] = []
    contrast_results: list[GammaPairwiseContrastResult] = []

    for g_index, gamma_R in enumerate(protocol.gamma_grid):
        for outcome in protocol.outcomes:
            family = _family_for(outcome)
            o_index = outcome_index[outcome]
            outcome_point = point_means[g_index, :, o_index]
            outcome_boot = bootstrap_means[:, g_index, :, o_index]

            for topology in protocol.topology_labels:
                t_index = topology_index[topology]
                lower, upper = _percentile_interval(
                    outcome_boot[:, t_index], protocol.confidence_level
                )
                mean_results.append(
                    GammaTopologyMeanResult(
                        family=family,
                        gamma_R=gamma_R,
                        outcome=outcome,
                        topology=topology,
                        estimate=float(outcome_point[t_index]),
                        ci_lower=lower,
                        ci_upper=upper,
                        n_replications=n_replications,
                    )
                )

            absolute_gap = float(np.max(outcome_point) - np.min(outcome_point))
            bootstrap_absolute_gap = np.max(outcome_boot, axis=1) - np.min(outcome_boot, axis=1)
            gap_lower, gap_upper = _percentile_interval(
                bootstrap_absolute_gap, protocol.confidence_level
            )

            relative_gap = None
            relative_gap_lower = None
            relative_gap_upper = None
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
                GammaTopologyGapResult(
                    family=family,
                    gamma_R=gamma_R,
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
                bootstrap_contrast = outcome_boot[:, left_index] - outcome_boot[:, right_index]
                lower, upper = _percentile_interval(
                    bootstrap_contrast, protocol.confidence_level
                )

                relative_effect = None
                relative_lower = None
                relative_upper = None
                if protocol.uses_relative_effect(outcome):
                    denominator = float(
                        0.5 * (outcome_point[left_index] + outcome_point[right_index])
                        + protocol.relative_epsilon
                    )
                    relative_effect = estimate / denominator
                    bootstrap_denominator = (
                        0.5 * (outcome_boot[:, left_index] + outcome_boot[:, right_index])
                        + protocol.relative_epsilon
                    )
                    bootstrap_relative = bootstrap_contrast / bootstrap_denominator
                    relative_lower, relative_upper = _percentile_interval(
                        bootstrap_relative, protocol.confidence_level
                    )

                contrast_results.append(
                    GammaPairwiseContrastResult(
                        family=family,
                        gamma_R=gamma_R,
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

    return GammaSweepAnalysisResult(
        experiment_seed=protocol.experiment_seed,
        bootstrap_seed=protocol.bootstrap_seed,
        n_replications=n_replications,
        n_bootstrap=protocol.n_bootstrap,
        confidence_level=protocol.confidence_level,
        alpha_anchor=protocol.alpha_anchor,
        beta_anchor=protocol.beta_anchor,
        gamma_grid=protocol.gamma_grid,
        cross_gamma_common_random_numbers_verified=True,
        topology_means=tuple(mean_results),
        topology_gaps=tuple(gap_results),
        pairwise_contrasts=tuple(contrast_results),
    )
