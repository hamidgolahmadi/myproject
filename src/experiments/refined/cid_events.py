"""Threshold-exceedance and operational-stabilisation diagnostics, Eqs. (247)-(250).

This module consumes an already-computed rolling CID path.  Thresholds and
component guardrails are explicit design inputs; none are inferred from the
realised path.  Missing guardrails are treated as inactive (+infinity), as
specified after Equation (250).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cid import RollingCIDPoint


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _optional_positive_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _positive_finite(name, value)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


@dataclass(frozen=True, slots=True)
class CIDThresholdConfiguration:
    """Pre-specified threshold and optional raw-component guardrails, Eq. (247)."""

    cid_threshold: float
    max_return_volatility: float | None = None
    max_belief_dispersion: float | None = None
    max_order_flow_pressure: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cid_threshold",
            _positive_finite("cid_threshold", self.cid_threshold),
        )
        for name in (
            "max_return_volatility",
            "max_belief_dispersion",
            "max_order_flow_pressure",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_finite(name, getattr(self, name)),
            )

    @property
    def return_limit(self) -> float:
        return np.inf if self.max_return_volatility is None else self.max_return_volatility

    @property
    def belief_limit(self) -> float:
        return np.inf if self.max_belief_dispersion is None else self.max_belief_dispersion

    @property
    def order_flow_limit(self) -> float:
        return np.inf if self.max_order_flow_pressure is None else self.max_order_flow_pressure


@dataclass(frozen=True, slots=True)
class OperationalStabilisationResult:
    """Equation (250) event result with explicit right censoring."""

    stabilisation_length: int
    stabilised: bool
    stabilisation_period: int | None
    right_censored: bool
    last_eligible_start_period: int | None

    def __post_init__(self) -> None:
        length = _positive_integer("stabilisation_length", self.stabilisation_length)
        object.__setattr__(self, "stabilisation_length", length)

        if not isinstance(self.stabilised, bool):
            raise TypeError("stabilised must be a bool")
        if not isinstance(self.right_censored, bool):
            raise TypeError("right_censored must be a bool")
        if self.right_censored == self.stabilised:
            raise ValueError("exactly one of stabilised and right_censored must be true")

        for name in ("stabilisation_period", "last_eligible_start_period"):
            value = getattr(self, name)
            if value is not None:
                value = _positive_integer(name, value)
                object.__setattr__(self, name, value)

        if self.stabilised and self.stabilisation_period is None:
            raise ValueError("stabilised result must contain stabilisation_period")
        if self.right_censored and self.stabilisation_period is not None:
            raise ValueError("right-censored result must not invent stabilisation_period")
        if (
            self.stabilisation_period is not None
            and self.last_eligible_start_period is not None
            and self.stabilisation_period > self.last_eligible_start_period
        ):
            raise ValueError("stabilisation_period cannot exceed last eligible start")


@dataclass(frozen=True, slots=True)
class CIDRunClassification:
    """Run-level diagnostics defined by Equations (247), (249), and (250)."""

    threshold_exceeding: bool
    peak_cid: float
    cid_exceedance_duration_share: float
    stabilisation: OperationalStabilisationResult

    def __post_init__(self) -> None:
        if not isinstance(self.threshold_exceeding, bool):
            raise TypeError("threshold_exceeding must be a bool")
        peak = float(self.peak_cid)
        duration = float(self.cid_exceedance_duration_share)
        if not np.isfinite(peak) or peak < 0.0:
            raise ValueError("peak_cid must be finite and non-negative")
        if not np.isfinite(duration) or not 0.0 <= duration <= 1.0:
            raise ValueError("cid_exceedance_duration_share must lie in [0,1]")
        if not isinstance(self.stabilisation, OperationalStabilisationResult):
            raise TypeError("stabilisation must be an OperationalStabilisationResult")
        object.__setattr__(self, "peak_cid", peak)
        object.__setattr__(self, "cid_exceedance_duration_share", duration)


def _validate_path(points: tuple[RollingCIDPoint, ...]) -> tuple[RollingCIDPoint, ...]:
    if not isinstance(points, tuple):
        raise TypeError("points must be a tuple of RollingCIDPoint")
    if len(points) == 0:
        raise ValueError("points must contain at least one rolling CID point")
    if not all(isinstance(point, RollingCIDPoint) for point in points):
        raise TypeError("points must contain only RollingCIDPoint objects")

    window_length = points[0].window_length
    previous_endpoint = points[0].endpoint_period - 1
    for point in points:
        if point.window_length != window_length:
            raise ValueError("all rolling CID points must use the same window_length")
        if point.endpoint_period != previous_endpoint + 1:
            raise ValueError("rolling CID endpoints must be consecutive periods")
        previous_endpoint = point.endpoint_period
    return points


def _outside_admissible_region(
    point: RollingCIDPoint,
    thresholds: CIDThresholdConfiguration,
) -> bool:
    return bool(
        point.cid > thresholds.cid_threshold
        or point.rolling_return_volatility > thresholds.return_limit
        or point.rolling_belief_dispersion > thresholds.belief_limit
        or point.rms_order_flow_pressure > thresholds.order_flow_limit
    )


def _inside_admissible_region(
    point: RollingCIDPoint,
    thresholds: CIDThresholdConfiguration,
) -> bool:
    return bool(
        point.cid <= thresholds.cid_threshold
        and point.rolling_return_volatility <= thresholds.return_limit
        and point.rolling_belief_dispersion <= thresholds.belief_limit
        and point.rms_order_flow_pressure <= thresholds.order_flow_limit
    )


def operational_stabilisation(
    points: tuple[RollingCIDPoint, ...],
    *,
    thresholds: CIDThresholdConfiguration,
    stabilisation_length: int = 50,
) -> OperationalStabilisationResult:
    """Return the first Eq. (250) stabilisation start, or a right-censored result.

    The report's first-stage protocol uses ``stabilisation_length=50``.  A
    qualifying start requires the CID and every active component guardrail to
    remain inside the admissible region for the complete consecutive block.
    """

    points = _validate_path(points)
    if not isinstance(thresholds, CIDThresholdConfiguration):
        raise TypeError("thresholds must be a CIDThresholdConfiguration")
    stabilisation_length = _positive_integer("stabilisation_length", stabilisation_length)

    if stabilisation_length > len(points):
        return OperationalStabilisationResult(
            stabilisation_length=stabilisation_length,
            stabilised=False,
            stabilisation_period=None,
            right_censored=True,
            last_eligible_start_period=None,
        )

    last_start_index = len(points) - stabilisation_length
    last_eligible_start_period = points[last_start_index].endpoint_period
    for start_index in range(last_start_index + 1):
        block = points[start_index : start_index + stabilisation_length]
        if all(_inside_admissible_region(point, thresholds) for point in block):
            return OperationalStabilisationResult(
                stabilisation_length=stabilisation_length,
                stabilised=True,
                stabilisation_period=points[start_index].endpoint_period,
                right_censored=False,
                last_eligible_start_period=last_eligible_start_period,
            )

    return OperationalStabilisationResult(
        stabilisation_length=stabilisation_length,
        stabilised=False,
        stabilisation_period=None,
        right_censored=True,
        last_eligible_start_period=last_eligible_start_period,
    )


def classify_cid_path(
    points: tuple[RollingCIDPoint, ...],
    *,
    thresholds: CIDThresholdConfiguration,
    stabilisation_length: int = 50,
) -> CIDRunClassification:
    """Compute run-level Eqs. (247), (249), and (250) from a rolling CID path."""

    points = _validate_path(points)
    if not isinstance(thresholds, CIDThresholdConfiguration):
        raise TypeError("thresholds must be a CIDThresholdConfiguration")

    threshold_exceeding = any(
        _outside_admissible_region(point, thresholds) for point in points
    )
    peak_cid = max(point.cid for point in points)
    cid_duration_share = float(
        np.mean([point.cid > thresholds.cid_threshold for point in points])
    )
    stabilisation = operational_stabilisation(
        points,
        thresholds=thresholds,
        stabilisation_length=stabilisation_length,
    )
    return CIDRunClassification(
        threshold_exceeding=threshold_exceeding,
        peak_cid=peak_cid,
        cid_exceedance_duration_share=cid_duration_share,
        stabilisation=stabilisation,
    )


def threshold_exceedance_rate(
    classifications: tuple[CIDRunClassification, ...],
) -> float:
    """Topology-level threshold-exceedance rate, Equation (248)."""

    if not isinstance(classifications, tuple):
        raise TypeError("classifications must be a tuple of CIDRunClassification")
    if len(classifications) == 0:
        raise ValueError("classifications must contain at least one run")
    if not all(isinstance(item, CIDRunClassification) for item in classifications):
        raise TypeError("classifications must contain only CIDRunClassification objects")
    return float(np.mean([item.threshold_exceeding for item in classifications]))
