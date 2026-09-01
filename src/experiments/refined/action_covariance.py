"""Rolling action-covariance diagnostics from Section 5.5, Eqs. (239)-(240).

This module evaluates an already-completed :class:`SimulationResult`.  It does
not rerun or alter the market transition.  For each valid rolling endpoint it
computes sample action covariances over the exact full post-burn-in window and
verifies the finite-sample variance decomposition for signed net order flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.refined import SimulationResult


_DECOMP_ATOL = 1e-12


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


def _validate_inputs(
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
        raise ValueError("window_length must be at least two for sample covariance")
    if window_length > n_post_burn:
        raise ValueError("window_length cannot exceed the post-burn-in sample length")

    n_agents = result.initial_state.n_agents
    if n_agents < 2:
        raise ValueError("action covariance requires at least two agents")
    if any(state.n_agents != n_agents for state in result.states):
        raise ValueError("all simulation states must have the same agent dimension")
    return window_length, burn_in, n_agents


@dataclass(frozen=True, slots=True)
class RollingActionCovariancePoint:
    """Equation (239)-(240) diagnostics at one rolling endpoint."""

    endpoint_period: int
    window_start_period: int
    window_length: int
    average_pairwise_action_covariance: float
    sum_individual_action_variances: float
    aggregate_order_flow_variance: float
    reconstructed_order_flow_variance: float

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
            "average_pairwise_action_covariance",
            "sum_individual_action_variances",
            "aggregate_order_flow_variance",
            "reconstructed_order_flow_variance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        if self.sum_individual_action_variances < -_DECOMP_ATOL:
            raise ValueError("sum_individual_action_variances cannot be negative")
        if self.aggregate_order_flow_variance < -_DECOMP_ATOL:
            raise ValueError("aggregate_order_flow_variance cannot be negative")
        if self.reconstructed_order_flow_variance < -_DECOMP_ATOL:
            raise ValueError("reconstructed_order_flow_variance cannot be negative")

    @property
    def decomposition_error(self) -> float:
        return float(self.aggregate_order_flow_variance - self.reconstructed_order_flow_variance)


def rolling_action_covariance(
    result: SimulationResult,
    *,
    window_length: int,
    burn_in: int = 0,
) -> tuple[RollingActionCovariancePoint, ...]:
    """Compute rolling Eqs. (239)-(240) over full post-burn-in windows.

    With report period labels ``t=1,...,T``, valid endpoints are
    ``t = burn_in + window_length, ..., T``.  Each covariance and variance is a
    finite-sample statistic with denominator ``window_length - 1``.
    """

    window_length, burn_in, n_agents = _validate_inputs(
        result,
        window_length=window_length,
        burn_in=burn_in,
    )

    actions = np.stack([output.actions for output in result.period_outputs], axis=0)
    flows = np.array([output.net_order_flow for output in result.period_outputs], dtype=float)

    if actions.shape != (result.n_periods, n_agents):
        raise ValueError("period action arrays have inconsistent dimensions")
    if not np.allclose(actions.sum(axis=1), flows, rtol=0.0, atol=_DECOMP_ATOL):
        raise ValueError("stored net order flow is inconsistent with the sum of agent actions")

    points: list[RollingActionCovariancePoint] = []
    first_endpoint_index = burn_in + window_length - 1
    for endpoint_index in range(first_endpoint_index, result.n_periods):
        start_index = endpoint_index - window_length + 1
        window_actions = actions[start_index : endpoint_index + 1]
        window_flows = flows[start_index : endpoint_index + 1]

        covariance_matrix = np.cov(window_actions, rowvar=False, ddof=1)
        if covariance_matrix.shape != (n_agents, n_agents):
            raise RuntimeError("unexpected covariance-matrix shape")

        upper_sum = float(np.sum(np.triu(covariance_matrix, k=1)))
        average_pairwise = 2.0 * upper_sum / (n_agents * (n_agents - 1))
        sum_individual_variances = float(np.trace(covariance_matrix))
        aggregate_variance = float(np.var(window_flows, ddof=1))
        reconstructed = float(
            sum_individual_variances
            + n_agents * (n_agents - 1) * average_pairwise
        )

        if not np.isclose(
            aggregate_variance,
            reconstructed,
            rtol=1e-10,
            atol=_DECOMP_ATOL,
        ):
            raise RuntimeError("Equation (240) sample variance decomposition failed")

        points.append(
            RollingActionCovariancePoint(
                endpoint_period=endpoint_index + 1,
                window_start_period=start_index + 1,
                window_length=window_length,
                average_pairwise_action_covariance=average_pairwise,
                sum_individual_action_variances=sum_individual_variances,
                aggregate_order_flow_variance=aggregate_variance,
                reconstructed_order_flow_variance=reconstructed,
            )
        )

    return tuple(points)
