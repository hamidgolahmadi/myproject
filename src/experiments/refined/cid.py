"""Dimensionless Composite Instability Diagnostic, Section 5.5 Eqs. (241)-(246).

This module evaluates an already-completed :class:`SimulationResult`.  It first
computes the three raw rolling components without calibration, then applies
explicit user-supplied positive reference scales and non-negative CID weights.
It never chooses scales from realised topology rankings and never reruns the
economic transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.refined import SimulationResult


_WEIGHT_ATOL = 1e-12


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_integer(name: str, value: int) -> int:
    value = _nonnegative_integer(name, value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


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


def _nonnegative_finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_rolling_inputs(
    result: SimulationResult,
    *,
    window_length: int,
    burn_in: int,
) -> tuple[int, int, int]:
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    window_length = _positive_integer("window_length", window_length)
    burn_in = _nonnegative_integer("burn_in", burn_in)
    if burn_in >= result.n_periods:
        raise ValueError("burn_in must satisfy burn_in < number of simulated periods")
    n_post_burn = result.n_periods - burn_in
    if window_length < 2:
        raise ValueError("window_length must be at least two for rolling return volatility")
    if window_length > n_post_burn:
        raise ValueError("window_length cannot exceed the post-burn-in sample length")

    n_agents = result.initial_state.n_agents
    if n_agents < 1:
        raise ValueError("simulation must contain at least one agent")
    if any(state.n_agents != n_agents for state in result.states):
        raise ValueError("all simulation states must have the same agent dimension")
    return window_length, burn_in, n_agents


@dataclass(frozen=True, slots=True)
class RollingCIDComponentsPoint:
    """Raw rolling components from Equations (241)-(243)."""

    endpoint_period: int
    window_start_period: int
    window_length: int
    rolling_return_volatility: float
    rolling_belief_dispersion: float
    rms_order_flow_pressure: float

    def __post_init__(self) -> None:
        for name in ("endpoint_period", "window_start_period", "window_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            value = int(value)
            if value < 1:
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, value)
        if self.window_start_period + self.window_length - 1 != self.endpoint_period:
            raise ValueError("window endpoints are inconsistent with window_length")

        for name in (
            "rolling_return_volatility",
            "rolling_belief_dispersion",
            "rms_order_flow_pressure",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_finite(name, getattr(self, name)),
            )


@dataclass(frozen=True, slots=True)
class CIDReferenceScales:
    """Positive pre-specified normalisation scales from Equation (244)."""

    return_scale: float
    belief_scale: float
    order_flow_scale: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "return_scale",
            _positive_finite("return_scale", self.return_scale),
        )
        object.__setattr__(
            self,
            "belief_scale",
            _positive_finite("belief_scale", self.belief_scale),
        )
        object.__setattr__(
            self,
            "order_flow_scale",
            _positive_finite("order_flow_scale", self.order_flow_scale),
        )


@dataclass(frozen=True, slots=True)
class CIDWeights:
    """Non-negative weights summing to one, Equation (245)."""

    return_weight: float
    belief_weight: float
    order_flow_weight: float

    def __post_init__(self) -> None:
        for name in ("return_weight", "belief_weight", "order_flow_weight"):
            object.__setattr__(
                self,
                name,
                _nonnegative_finite(name, getattr(self, name)),
            )
        total = self.return_weight + self.belief_weight + self.order_flow_weight
        if not np.isclose(total, 1.0, rtol=0.0, atol=_WEIGHT_ATOL):
            raise ValueError("CID weights must sum to one")

    @classmethod
    def equal(cls) -> "CIDWeights":
        """Return the transparent equal-weight baseline mentioned after Eq. (246)."""

        return cls(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


@dataclass(frozen=True, slots=True)
class RollingCIDPoint:
    """Standardised components and dimensionless CID at one endpoint."""

    endpoint_period: int
    window_start_period: int
    window_length: int
    rolling_return_volatility: float
    rolling_belief_dispersion: float
    rms_order_flow_pressure: float
    standardised_return: float
    standardised_belief: float
    standardised_order_flow: float
    cid: float

    def __post_init__(self) -> None:
        for name in ("endpoint_period", "window_start_period", "window_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            value = int(value)
            if value < 1:
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, value)
        if self.window_start_period + self.window_length - 1 != self.endpoint_period:
            raise ValueError("window endpoints are inconsistent with window_length")

        for name in (
            "rolling_return_volatility",
            "rolling_belief_dispersion",
            "rms_order_flow_pressure",
            "standardised_return",
            "standardised_belief",
            "standardised_order_flow",
            "cid",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_finite(name, getattr(self, name)),
            )


def rolling_cid_components(
    result: SimulationResult,
    *,
    window_length: int,
    burn_in: int = 0,
) -> tuple[RollingCIDComponentsPoint, ...]:
    """Compute raw rolling components in Equations (241)-(243).

    Report periods are labelled ``t=1,...,T``.  Valid endpoints are
    ``t = burn_in + window_length, ..., T`` and every window contains exactly
    ``window_length`` post-burn-in observations as required by Equation (235).

    ``D^b_u`` in Equation (242) is implemented as the population
    cross-sectional belief variance, consistent with the explicit definition
    used for the report's belief-variance outcome in Equation (289).
    """

    window_length, burn_in, n_agents = _validate_rolling_inputs(
        result,
        window_length=window_length,
        burn_in=burn_in,
    )

    returns = np.array([output.return_ for output in result.period_outputs], dtype=float)
    flows = np.array([output.net_order_flow for output in result.period_outputs], dtype=float)
    belief_dispersion = np.array(
        [np.var(state.beliefs, ddof=0) for state in result.states[1:]],
        dtype=float,
    )

    if returns.shape != (result.n_periods,):
        raise RuntimeError("unexpected return array shape")
    if flows.shape != (result.n_periods,):
        raise RuntimeError("unexpected order-flow array shape")
    if belief_dispersion.shape != (result.n_periods,):
        raise RuntimeError("unexpected belief-dispersion array shape")

    points: list[RollingCIDComponentsPoint] = []
    first_endpoint_index = burn_in + window_length - 1
    for endpoint_index in range(first_endpoint_index, result.n_periods):
        start_index = endpoint_index - window_length + 1
        window_returns = returns[start_index : endpoint_index + 1]
        window_beliefs = belief_dispersion[start_index : endpoint_index + 1]
        window_flows = flows[start_index : endpoint_index + 1]

        return_volatility = float(np.std(window_returns, ddof=1))
        belief_component = float(np.mean(window_beliefs))
        order_flow_pressure = float(
            np.sqrt(np.mean(np.square(window_flows / n_agents)))
        )

        points.append(
            RollingCIDComponentsPoint(
                endpoint_period=endpoint_index + 1,
                window_start_period=start_index + 1,
                window_length=window_length,
                rolling_return_volatility=return_volatility,
                rolling_belief_dispersion=belief_component,
                rms_order_flow_pressure=order_flow_pressure,
            )
        )

    return tuple(points)


def standardise_cid_components(
    components: tuple[RollingCIDComponentsPoint, ...],
    *,
    scales: CIDReferenceScales,
    weights: CIDWeights,
) -> tuple[RollingCIDPoint, ...]:
    """Apply Equations (244)-(246) to already-computed raw components."""

    if not isinstance(scales, CIDReferenceScales):
        raise TypeError("scales must be a CIDReferenceScales")
    if not isinstance(weights, CIDWeights):
        raise TypeError("weights must be a CIDWeights")
    if not isinstance(components, tuple):
        raise TypeError("components must be a tuple of RollingCIDComponentsPoint")
    if len(components) == 0:
        raise ValueError("components must contain at least one rolling point")
    if not all(isinstance(point, RollingCIDComponentsPoint) for point in components):
        raise TypeError("components must contain only RollingCIDComponentsPoint objects")

    result: list[RollingCIDPoint] = []
    for point in components:
        z_return = point.rolling_return_volatility / scales.return_scale
        z_belief = point.rolling_belief_dispersion / scales.belief_scale
        z_flow = point.rms_order_flow_pressure / scales.order_flow_scale
        cid_value = (
            weights.return_weight * z_return
            + weights.belief_weight * z_belief
            + weights.order_flow_weight * z_flow
        )
        result.append(
            RollingCIDPoint(
                endpoint_period=point.endpoint_period,
                window_start_period=point.window_start_period,
                window_length=point.window_length,
                rolling_return_volatility=point.rolling_return_volatility,
                rolling_belief_dispersion=point.rolling_belief_dispersion,
                rms_order_flow_pressure=point.rms_order_flow_pressure,
                standardised_return=z_return,
                standardised_belief=z_belief,
                standardised_order_flow=z_flow,
                cid=cid_value,
            )
        )
    return tuple(result)


def rolling_cid(
    result: SimulationResult,
    *,
    window_length: int,
    scales: CIDReferenceScales,
    weights: CIDWeights,
    burn_in: int = 0,
) -> tuple[RollingCIDPoint, ...]:
    """Compute Equations (241)-(246) from one completed simulation path."""

    components = rolling_cid_components(
        result,
        window_length=window_length,
        burn_in=burn_in,
    )
    return standardise_cid_components(
        components,
        scales=scales,
        weights=weights,
    )
