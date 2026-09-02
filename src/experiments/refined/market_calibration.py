"""Pre-topology market-evaluation calibration protocol.

The doctoral report requires rolling-window lengths, CID reference scales,
weights, thresholds, and guardrails to be fixed before topology-evaluation
replications are inspected.  It also permits the reference scales to be
estimated under the no-social benchmark (alpha=0) using calibration seeds that
are not reused for topology evaluation.

This module implements a conservative two-sample calibration design:

1. a scale sample estimates c_ret, c_bel, and c_F;
2. an independent threshold sample, using the already-fixed scales, estimates
   the operational CID threshold from run-level peak CID values.

The module consumes already-computed rolling CID component paths.  It does not
simulate the market, choose economic parameters, or inspect topology rankings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .cid import (
    CIDReferenceScales,
    CIDWeights,
    RollingCIDComponentsPoint,
    standardise_cid_components,
)


_QUANTILE_METHOD = "higher"


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_probability(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    return value


@dataclass(frozen=True, slots=True)
class MarketEvaluationCalibrationProtocol:
    """Frozen design inputs for the first refined market calibration.

    The baseline choices below are explicit design decisions rather than
    report equations, except for ``burn_in=0`` and ``stabilisation_length=50``
    which are the report's first-stage choices.

    ``rolling_window=50`` is selected as a 5% window of the T=1000 first-stage
    horizon and is paired with pre-specified robustness windows 25 and 100.
    The two no-social calibration samples use disjoint reproducibility
    namespaces and must also remain disjoint from later topology-evaluation
    seeds.
    """

    scale_calibration_seed: int = 2026090201
    threshold_calibration_seed: int = 2026090202
    n_scale_replications: int = 500
    n_threshold_replications: int = 500
    horizon: int = 1000
    burn_in: int = 0
    rolling_window: int = 50
    calibration_alpha: float = 0.0
    cid_weights: CIDWeights = field(default_factory=CIDWeights.equal)
    cid_peak_quantile: float = 0.95
    stabilisation_length: int = 50

    def __post_init__(self) -> None:
        for name in (
            "scale_calibration_seed",
            "threshold_calibration_seed",
        ):
            value = _nonnegative_integer(name, getattr(self, name))
            object.__setattr__(self, name, value)
        if self.scale_calibration_seed == self.threshold_calibration_seed:
            raise ValueError("scale and threshold calibration seeds must be distinct")

        for name in (
            "n_scale_replications",
            "n_threshold_replications",
            "horizon",
            "rolling_window",
            "stabilisation_length",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(name, getattr(self, name)),
            )
        object.__setattr__(self, "burn_in", _nonnegative_integer("burn_in", self.burn_in))
        if self.burn_in >= self.horizon:
            raise ValueError("burn_in must be smaller than horizon")
        if self.rolling_window < 2:
            raise ValueError("rolling_window must be at least two")
        if self.rolling_window > self.horizon - self.burn_in:
            raise ValueError("rolling_window cannot exceed the post-burn-in horizon")

        if isinstance(self.calibration_alpha, bool):
            raise TypeError("calibration_alpha must be a real number")
        alpha = float(self.calibration_alpha)
        if not np.isfinite(alpha) or alpha != 0.0:
            raise ValueError("the calibration protocol requires the no-social benchmark alpha=0")
        object.__setattr__(self, "calibration_alpha", alpha)

        if not isinstance(self.cid_weights, CIDWeights):
            raise TypeError("cid_weights must be a CIDWeights")
        object.__setattr__(
            self,
            "cid_peak_quantile",
            _finite_probability("cid_peak_quantile", self.cid_peak_quantile),
        )

    @property
    def expected_rolling_points_per_run(self) -> int:
        """Number of full rolling endpoints implied by Eq. (235)."""

        return self.horizon - self.burn_in - self.rolling_window + 1

    @property
    def robustness_windows(self) -> tuple[int, int]:
        """Pre-specified rolling-window robustness values around the baseline."""

        return (25, 100)


@dataclass(frozen=True, slots=True)
class MarketEvaluationCalibration:
    """Frozen numerical CID calibration produced before topology evaluation."""

    protocol: MarketEvaluationCalibrationProtocol
    reference_scales: CIDReferenceScales
    cid_weights: CIDWeights
    cid_threshold: float
    return_guardrail: None = None
    belief_guardrail: None = None
    order_flow_guardrail: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, MarketEvaluationCalibrationProtocol):
            raise TypeError("protocol must be a MarketEvaluationCalibrationProtocol")
        if not isinstance(self.reference_scales, CIDReferenceScales):
            raise TypeError("reference_scales must be a CIDReferenceScales")
        if not isinstance(self.cid_weights, CIDWeights):
            raise TypeError("cid_weights must be a CIDWeights")
        threshold = float(self.cid_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("cid_threshold must be finite and strictly positive")
        object.__setattr__(self, "cid_threshold", threshold)
        if self.cid_weights != self.protocol.cid_weights:
            raise ValueError("calibration weights must match the frozen protocol weights")


ComponentPath = tuple[RollingCIDComponentsPoint, ...]


def first_market_evaluation_calibration_protocol() -> MarketEvaluationCalibrationProtocol:
    """Return the first explicit refined market-calibration design."""

    return MarketEvaluationCalibrationProtocol()


def _normalise_paths(
    paths: Iterable[ComponentPath],
    *,
    expected_replications: int,
    protocol: MarketEvaluationCalibrationProtocol,
    sample_name: str,
) -> tuple[ComponentPath, ...]:
    try:
        path_tuple = tuple(paths)
    except TypeError as exc:
        raise TypeError(f"{sample_name} must be an iterable of component paths") from exc
    if len(path_tuple) != expected_replications:
        raise ValueError(
            f"{sample_name} must contain exactly {expected_replications} replications"
        )

    expected_points = protocol.expected_rolling_points_per_run
    for replication_index, path in enumerate(path_tuple):
        if not isinstance(path, tuple):
            raise TypeError(f"{sample_name} paths must be tuples")
        if len(path) != expected_points:
            raise ValueError(
                f"{sample_name} replication {replication_index} must contain "
                f"exactly {expected_points} rolling points"
            )
        if not all(isinstance(point, RollingCIDComponentsPoint) for point in path):
            raise TypeError(
                f"{sample_name} paths must contain only RollingCIDComponentsPoint objects"
            )
        expected_first_endpoint = protocol.burn_in + protocol.rolling_window
        for offset, point in enumerate(path):
            if point.window_length != protocol.rolling_window:
                raise ValueError(f"{sample_name} uses an unexpected rolling-window length")
            if point.endpoint_period != expected_first_endpoint + offset:
                raise ValueError(f"{sample_name} rolling endpoints must be consecutive")
    return path_tuple


def estimate_reference_scales(
    scale_paths: Iterable[ComponentPath],
    *,
    protocol: MarketEvaluationCalibrationProtocol,
) -> CIDReferenceScales:
    """Estimate Eq. (244) scales from the independent no-social scale sample.

    The baseline statistic is the pooled median of each raw rolling component.
    Medians make the scale definition resistant to a small number of extreme
    calibration windows.  A zero median is treated as a failed calibration,
    not silently repaired with an epsilon.
    """

    if not isinstance(protocol, MarketEvaluationCalibrationProtocol):
        raise TypeError("protocol must be a MarketEvaluationCalibrationProtocol")
    paths = _normalise_paths(
        scale_paths,
        expected_replications=protocol.n_scale_replications,
        protocol=protocol,
        sample_name="scale_paths",
    )
    points = tuple(point for path in paths for point in path)

    return_scale = float(np.median([point.rolling_return_volatility for point in points]))
    belief_scale = float(np.median([point.rolling_belief_dispersion for point in points]))
    flow_scale = float(np.median([point.rms_order_flow_pressure for point in points]))

    if min(return_scale, belief_scale, flow_scale) <= 0.0:
        raise ValueError(
            "pooled-median calibration produced a non-positive reference scale; "
            "the market specification must be reviewed rather than patched with an epsilon"
        )
    return CIDReferenceScales(
        return_scale=return_scale,
        belief_scale=belief_scale,
        order_flow_scale=flow_scale,
    )


def estimate_cid_threshold(
    threshold_paths: Iterable[ComponentPath],
    *,
    protocol: MarketEvaluationCalibrationProtocol,
    scales: CIDReferenceScales,
) -> float:
    """Estimate c_CID from an independent no-social threshold sample.

    Each calibration run is first reduced to its *peak* CID.  The threshold is
    then the pre-specified empirical quantile of those run-level peaks.  This
    targets an operational run-level exceedance rate rather than a pointwise
    window exceedance rate.  NumPy's ``higher`` quantile convention is used for
    a deterministic conservative finite-sample cutoff.
    """

    if not isinstance(protocol, MarketEvaluationCalibrationProtocol):
        raise TypeError("protocol must be a MarketEvaluationCalibrationProtocol")
    if not isinstance(scales, CIDReferenceScales):
        raise TypeError("scales must be a CIDReferenceScales")
    paths = _normalise_paths(
        threshold_paths,
        expected_replications=protocol.n_threshold_replications,
        protocol=protocol,
        sample_name="threshold_paths",
    )

    peaks = []
    for path in paths:
        cid_path = standardise_cid_components(
            path,
            scales=scales,
            weights=protocol.cid_weights,
        )
        peaks.append(max(point.cid for point in cid_path))

    threshold = float(
        np.quantile(
            np.asarray(peaks, dtype=float),
            protocol.cid_peak_quantile,
            method=_QUANTILE_METHOD,
        )
    )
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold calibration produced a non-positive CID threshold")
    return threshold


def calibrate_market_evaluation(
    scale_paths: Iterable[ComponentPath],
    threshold_paths: Iterable[ComponentPath],
    *,
    protocol: MarketEvaluationCalibrationProtocol,
) -> MarketEvaluationCalibration:
    """Produce the first-stage CID calibration without inspecting topology outcomes."""

    scales = estimate_reference_scales(scale_paths, protocol=protocol)
    threshold = estimate_cid_threshold(
        threshold_paths,
        protocol=protocol,
        scales=scales,
    )
    return MarketEvaluationCalibration(
        protocol=protocol,
        reference_scales=scales,
        cid_weights=protocol.cid_weights,
        cid_threshold=threshold,
    )
