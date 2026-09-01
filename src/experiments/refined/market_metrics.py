"""Run-level refined market outcomes from Section 5.5, Eqs. (231)-(238), (288)-(289).

This module evaluates an already-completed :class:`SimulationResult`.  It does
not rerun, alter, or duplicate any economic transition equation.  Period
``t=1,...,T`` corresponds to ``period_outputs[t-1]`` and ``states[t]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.refined import SimulationResult


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _evaluation_slice(result: SimulationResult, burn_in: int) -> slice:
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    burn_in = _nonnegative_integer("burn_in", burn_in)
    if burn_in >= result.n_periods:
        raise ValueError("burn_in must satisfy burn_in < number of simulated periods")
    return slice(burn_in, result.n_periods)


def _n_agents(result: SimulationResult) -> int:
    n_agents = result.initial_state.n_agents
    if any(state.n_agents != n_agents for state in result.states):
        raise ValueError("all simulation states must have the same agent dimension")
    return n_agents


def _returns(result: SimulationResult, evaluation: slice) -> np.ndarray:
    return np.array(
        [output.return_ for output in result.period_outputs[evaluation]],
        dtype=float,
    )


def _mispricing(result: SimulationResult, evaluation: slice) -> np.ndarray:
    start = 0 if evaluation.start is None else int(evaluation.start)
    stop = result.n_periods if evaluation.stop is None else int(evaluation.stop)
    prices = np.array([state.price for state in result.states[1 + start : 1 + stop]], dtype=float)
    fundamentals = np.array(
        [output.fundamental_value for output in result.period_outputs[start:stop]],
        dtype=float,
    )
    return prices - fundamentals


def return_volatility(result: SimulationResult, *, burn_in: int = 0) -> float:
    """Sample standard deviation of returns on ``T_B``, Equation (236)."""

    evaluation = _evaluation_slice(result, burn_in)
    returns = _returns(result, evaluation)
    if returns.size < 2:
        raise ValueError("return volatility requires at least two evaluated periods")
    return float(np.std(returns, ddof=1))


def rms_mispricing(result: SimulationResult, *, burn_in: int = 0) -> float:
    """Root-mean-square price mispricing on ``T_B``, Equation (237)."""

    evaluation = _evaluation_slice(result, burn_in)
    errors = _mispricing(result, evaluation)
    return float(np.sqrt(np.mean(np.square(errors))))


def maximum_absolute_mispricing(
    result: SimulationResult,
    *,
    burn_in: int = 0,
) -> float:
    """Maximum absolute price mispricing on ``T_B``, Equation (237)."""

    evaluation = _evaluation_slice(result, burn_in)
    return float(np.max(np.abs(_mispricing(result, evaluation))))


def mean_absolute_order_flow_per_agent(
    result: SimulationResult,
    *,
    burn_in: int = 0,
) -> float:
    """Mean absolute signed net order flow per agent, Equation (238)."""

    evaluation = _evaluation_slice(result, burn_in)
    n_agents = _n_agents(result)
    flows = np.array(
        [output.net_order_flow for output in result.period_outputs[evaluation]],
        dtype=float,
    )
    return float(np.mean(np.abs(flows) / n_agents))


def mean_absolute_return(result: SimulationResult, *, burn_in: int = 0) -> float:
    """Run-level mean absolute return, Equation (288)."""

    evaluation = _evaluation_slice(result, burn_in)
    return float(np.mean(np.abs(_returns(result, evaluation))))


def time_averaged_belief_variance(
    result: SimulationResult,
    *,
    burn_in: int = 0,
) -> float:
    """Time-averaged cross-sectional population belief variance, Equation (289)."""

    evaluation = _evaluation_slice(result, burn_in)
    _n_agents(result)
    start = 0 if evaluation.start is None else int(evaluation.start)
    stop = result.n_periods if evaluation.stop is None else int(evaluation.stop)
    variances = np.array(
        [np.var(state.beliefs, ddof=0) for state in result.states[1 + start : 1 + stop]],
        dtype=float,
    )
    return float(np.mean(variances))


@dataclass(frozen=True, slots=True)
class RunLevelMarketOutcomes:
    """Principal run-level outcome bundle for one completed simulation."""

    burn_in: int
    n_observations: int
    return_volatility: float
    rms_mispricing: float
    maximum_absolute_mispricing: float
    mean_absolute_order_flow_per_agent: float
    mean_absolute_return: float
    time_averaged_belief_variance: float

    def __post_init__(self) -> None:
        burn_in = _nonnegative_integer("burn_in", self.burn_in)
        if isinstance(self.n_observations, bool) or not isinstance(
            self.n_observations, (int, np.integer)
        ):
            raise TypeError("n_observations must be an integer")
        n_observations = int(self.n_observations)
        if n_observations < 1:
            raise ValueError("n_observations must be strictly positive")

        for name in (
            "return_volatility",
            "rms_mispricing",
            "maximum_absolute_mispricing",
            "mean_absolute_order_flow_per_agent",
            "mean_absolute_return",
            "time_averaged_belief_variance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

        object.__setattr__(self, "burn_in", burn_in)
        object.__setattr__(self, "n_observations", n_observations)


def compute_run_level_market_outcomes(
    result: SimulationResult,
    *,
    burn_in: int = 0,
) -> RunLevelMarketOutcomes:
    """Compute Eqs. (236)-(238), (288)-(289) from one simulation path.

    The baseline report specification uses ``burn_in=0``.  Positive values are
    explicit robustness choices and are never inferred from the realised data.
    """

    evaluation = _evaluation_slice(result, burn_in)
    start = 0 if evaluation.start is None else int(evaluation.start)
    stop = result.n_periods if evaluation.stop is None else int(evaluation.stop)
    n_observations = stop - start
    if n_observations < 2:
        raise ValueError("run-level outcome bundle requires at least two evaluated periods")

    return RunLevelMarketOutcomes(
        burn_in=burn_in,
        n_observations=n_observations,
        return_volatility=return_volatility(result, burn_in=burn_in),
        rms_mispricing=rms_mispricing(result, burn_in=burn_in),
        maximum_absolute_mispricing=maximum_absolute_mispricing(
            result,
            burn_in=burn_in,
        ),
        mean_absolute_order_flow_per_agent=mean_absolute_order_flow_per_agent(
            result,
            burn_in=burn_in,
        ),
        mean_absolute_return=mean_absolute_return(result, burn_in=burn_in),
        time_averaged_belief_variance=time_averaged_belief_variance(
            result,
            burn_in=burn_in,
        ),
    )
