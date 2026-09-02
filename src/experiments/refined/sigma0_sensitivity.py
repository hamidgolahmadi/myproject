"""Controlled pre-freeze sensitivity smoke for the reputation floor ``sigma_0``.

The first baseline scale smoke showed that the provisional ``sigma_0=1e-6``
is tiny relative to realised local reputation dispersion.  Equation (58) uses
``sigma_0`` specifically to prevent an almost-degenerate local reputation
distribution from producing an artificial response after standardisation.

This module therefore performs a deliberately narrow OAT smoke before the
baseline is frozen.  All graph, shock, and initial-state randomness is kept
common across candidate ``sigma_0`` values.  Results are pooled across topology
labels and are for absolute scale/non-degeneracy assessment only; the routine
does not estimate topology treatment effects or rank R/SW/SF.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .baseline_specification import RefinedBaselineCandidate, first_refined_baseline_candidate
from .market_smoke import (
    MarketScaleSmokeProtocol,
    MarketScaleSmokeResult,
    SmokeMetricSummary,
    run_first_refined_baseline_scale_smoke,
)
from .seeding import nonnegative_integer


_DEFAULT_SIGMA0_VALUES = (1e-6, 1e-4, 5e-4, 1e-3, 2e-3)
_REPORTED_METRICS = (
    "return_std",
    "rms_mispricing",
    "rms_flow_per_agent",
    "desired_action_abs_p95",
    "execution_projection_fraction",
    "inventory_boundary_fraction",
    "median_local_reputation_std",
    "median_reputation_scale_to_sigma0",
    "mean_attention_mobility",
    "max_attention_mobility",
    "final_attention_distance_from_initial",
)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _sigma0_values(values: Iterable[float]) -> tuple[float, ...]:
    try:
        value_tuple = tuple(values)
    except TypeError as exc:
        raise TypeError("sigma0_values must be an iterable") from exc
    if len(value_tuple) < 2:
        raise ValueError("sigma0 sensitivity requires at least two candidate values")

    normalised: list[float] = []
    for value in value_tuple:
        if isinstance(value, bool):
            raise TypeError("sigma0 values must be real numbers")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("sigma0 values must be real numbers") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("sigma0 values must be finite and strictly positive")
        normalised.append(value)

    if len(set(normalised)) != len(normalised):
        raise ValueError("sigma0 sensitivity values must be unique")
    return tuple(normalised)


@dataclass(frozen=True, slots=True)
class Sigma0SensitivityProtocol:
    """Common-random-number OAT design for the pre-freeze ``sigma_0`` check."""

    experiment_seed: int = 2026090203
    n_replications: int = 5
    sigma0_values: tuple[float, ...] = _DEFAULT_SIGMA0_VALUES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_seed",
            nonnegative_integer("experiment_seed", self.experiment_seed),
        )
        object.__setattr__(
            self,
            "n_replications",
            _positive_integer("n_replications", self.n_replications),
        )
        object.__setattr__(self, "sigma0_values", _sigma0_values(self.sigma0_values))


@dataclass(frozen=True, slots=True)
class Sigma0SensitivityRow:
    """One pooled diagnostic row for one ``sigma_0`` and one metric."""

    sigma_0: float
    metric: str
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        sigma_0 = float(self.sigma_0)
        if not np.isfinite(sigma_0) or sigma_0 <= 0.0:
            raise ValueError("sigma_0 must be finite and strictly positive")
        if self.metric not in _REPORTED_METRICS:
            raise ValueError("unknown sigma0 sensitivity metric")
        count = _positive_integer("count", self.count)
        values = (self.mean, self.median, self.minimum, self.maximum)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("summary statistics must be finite")
        object.__setattr__(self, "sigma_0", sigma_0)
        object.__setattr__(self, "count", count)
        for name in ("mean", "median", "minimum", "maximum"):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class Sigma0SensitivityResult:
    """Full OAT smoke outputs, including reusable per-value smoke results."""

    protocol: Sigma0SensitivityProtocol
    candidate: RefinedBaselineCandidate
    smoke_results: tuple[MarketScaleSmokeResult, ...]
    pooled_rows: tuple[Sigma0SensitivityRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, Sigma0SensitivityProtocol):
            raise TypeError("protocol must be Sigma0SensitivityProtocol")
        if not isinstance(self.candidate, RefinedBaselineCandidate):
            raise TypeError("candidate must be RefinedBaselineCandidate")
        if len(self.smoke_results) != len(self.protocol.sigma0_values):
            raise ValueError("one smoke result is required for each sigma0 value")
        expected_rows = len(self.protocol.sigma0_values) * len(_REPORTED_METRICS)
        if len(self.pooled_rows) != expected_rows:
            raise ValueError(f"pooled_rows must contain exactly {expected_rows} rows")

    def rows_for(self, sigma_0: float) -> tuple[Sigma0SensitivityRow, ...]:
        value = float(sigma_0)
        rows = tuple(row for row in self.pooled_rows if row.sigma_0 == value)
        if not rows:
            raise KeyError(f"unknown sigma_0 value: {sigma_0}")
        return rows

    def metric_values(self, metric: str) -> tuple[tuple[float, float], ...]:
        if metric not in _REPORTED_METRICS:
            raise KeyError(f"unknown metric: {metric}")
        return tuple((row.sigma_0, row.median) for row in self.pooled_rows if row.metric == metric)


def _selected_rows(
    sigma_0: float,
    summaries: tuple[SmokeMetricSummary, ...],
) -> tuple[Sigma0SensitivityRow, ...]:
    summary_by_metric = {summary.metric: summary for summary in summaries}
    rows = []
    for metric in _REPORTED_METRICS:
        summary = summary_by_metric[metric]
        rows.append(
            Sigma0SensitivityRow(
                sigma_0=sigma_0,
                metric=metric,
                count=summary.count,
                mean=summary.mean,
                median=summary.median,
                minimum=summary.minimum,
                maximum=summary.maximum,
            )
        )
    return tuple(rows)


def run_sigma0_sensitivity_smoke(
    *,
    protocol: Sigma0SensitivityProtocol | None = None,
    candidate: RefinedBaselineCandidate | None = None,
) -> Sigma0SensitivityResult:
    """Run the common-random-number OAT smoke without topology contrasts."""

    if protocol is None:
        protocol = Sigma0SensitivityProtocol()
    if candidate is None:
        candidate = first_refined_baseline_candidate()
    if not isinstance(protocol, Sigma0SensitivityProtocol):
        raise TypeError("protocol must be Sigma0SensitivityProtocol")
    if not isinstance(candidate, RefinedBaselineCandidate):
        raise TypeError("candidate must be RefinedBaselineCandidate")

    smoke_protocol = MarketScaleSmokeProtocol(
        experiment_seed=protocol.experiment_seed,
        n_replications=protocol.n_replications,
    )

    smoke_results: list[MarketScaleSmokeResult] = []
    pooled_rows: list[Sigma0SensitivityRow] = []

    for sigma_0 in protocol.sigma0_values:
        parameters = replace(candidate.parameters, sigma_0=sigma_0)
        sigma_candidate = replace(candidate, parameters=parameters)
        smoke = run_first_refined_baseline_scale_smoke(
            protocol=smoke_protocol,
            candidate=sigma_candidate,
        )
        smoke_results.append(smoke)
        pooled_rows.extend(_selected_rows(sigma_0, smoke.pooled_summary))

    return Sigma0SensitivityResult(
        protocol=protocol,
        candidate=candidate,
        smoke_results=tuple(smoke_results),
        pooled_rows=tuple(pooled_rows),
    )
