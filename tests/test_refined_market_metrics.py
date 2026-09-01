"""Tests for report-defined refined run-level market outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.refined import (
    RunLevelMarketOutcomes,
    compute_run_level_market_outcomes,
    maximum_absolute_mispricing,
    mean_absolute_order_flow_per_agent,
    mean_absolute_return,
    return_volatility,
    rms_mispricing,
    time_averaged_belief_variance,
)
from src.model.refined import PeriodOutputs, RefinedState, SimulationResult


def _state(price: float, beliefs: list[float]) -> RefinedState:
    return RefinedState(
        theta=0.0,
        beliefs=np.array(beliefs, dtype=float),
        positions=np.zeros(2),
        price=price,
        reputation=np.zeros(2),
        attention=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )


def _output(
    *,
    fundamental: float,
    flow: float,
    return_: float,
    actions: tuple[float, float],
) -> PeriodOutputs:
    zeros = np.zeros(2)
    return PeriodOutputs(
        fundamental_value=fundamental,
        signals=zeros,
        perceived_values=zeros,
        valuation_gaps=zeros,
        desired_actions=np.array(actions, dtype=float),
        actions=np.array(actions, dtype=float),
        net_order_flow=flow,
        return_=return_,
        profits=zeros,
        reputation_scores=np.zeros((2, 2)),
    )


def example_result() -> SimulationResult:
    """Four-period path with hand-checkable report metrics.

    returns:      [ 1, -1,  3,  1]
    prices:       [101,100,103,104]
    fundamentals: [100,101,102,103]
    mispricing:   [ 1, -1,  1,  1]
    net flows:    [ 2, -4,  0,  6]

    At t=3 the actions are large and offset exactly; this distinguishes signed
    net order flow from gross trading volume.
    """

    states = (
        _state(100.0, [0.0, 0.0]),
        _state(101.0, [0.0, 2.0]),
        _state(100.0, [1.0, 1.0]),
        _state(103.0, [-1.0, 3.0]),
        _state(104.0, [2.0, 4.0]),
    )
    outputs = (
        _output(fundamental=100.0, flow=2.0, return_=1.0, actions=(1.0, 1.0)),
        _output(fundamental=101.0, flow=-4.0, return_=-1.0, actions=(-2.0, -2.0)),
        _output(fundamental=102.0, flow=0.0, return_=3.0, actions=(5.0, -5.0)),
        _output(fundamental=103.0, flow=6.0, return_=1.0, actions=(3.0, 3.0)),
    )
    return SimulationResult(states=states, period_outputs=outputs)


def test_return_volatility_matches_equation_236_with_sample_denominator() -> None:
    expected = np.sqrt(8.0 / 3.0)
    assert return_volatility(example_result()) == pytest.approx(expected)


def test_rms_mispricing_uses_contemporaneous_price_and_fundamental() -> None:
    assert rms_mispricing(example_result()) == pytest.approx(1.0)


def test_maximum_absolute_mispricing_matches_equation_237() -> None:
    assert maximum_absolute_mispricing(example_result()) == pytest.approx(1.0)


def test_mean_absolute_order_flow_uses_signed_net_flow_not_gross_volume() -> None:
    # |F_t|/N = [1, 2, 0, 3], whose mean is 1.5.  The cancelling t=3
    # actions (5,-5) must contribute zero rather than gross volume 10.
    assert mean_absolute_order_flow_per_agent(example_result()) == pytest.approx(1.5)


def test_mean_absolute_return_matches_equation_288() -> None:
    assert mean_absolute_return(example_result()) == pytest.approx(1.5)


def test_time_averaged_belief_variance_uses_population_cross_section() -> None:
    # Cross-sectional variances are [1, 0, 4, 1].
    assert time_averaged_belief_variance(example_result()) == pytest.approx(1.5)


def test_positive_burn_in_selects_exactly_periods_b_plus_1_through_t() -> None:
    result = example_result()
    assert return_volatility(result, burn_in=1) == pytest.approx(2.0)
    assert rms_mispricing(result, burn_in=1) == pytest.approx(1.0)
    assert maximum_absolute_mispricing(result, burn_in=1) == pytest.approx(1.0)
    assert mean_absolute_order_flow_per_agent(result, burn_in=1) == pytest.approx(5.0 / 3.0)
    assert mean_absolute_return(result, burn_in=1) == pytest.approx(5.0 / 3.0)
    assert time_averaged_belief_variance(result, burn_in=1) == pytest.approx(5.0 / 3.0)


def test_outcome_bundle_matches_individual_metrics_and_reports_sample_size() -> None:
    result = example_result()
    outcomes = compute_run_level_market_outcomes(result, burn_in=1)

    assert outcomes.burn_in == 1
    assert outcomes.n_observations == 3
    assert outcomes.return_volatility == pytest.approx(2.0)
    assert outcomes.rms_mispricing == pytest.approx(1.0)
    assert outcomes.maximum_absolute_mispricing == pytest.approx(1.0)
    assert outcomes.mean_absolute_order_flow_per_agent == pytest.approx(5.0 / 3.0)
    assert outcomes.mean_absolute_return == pytest.approx(5.0 / 3.0)
    assert outcomes.time_averaged_belief_variance == pytest.approx(5.0 / 3.0)


@pytest.mark.parametrize("burn_in", [-1, 4, 5])
def test_metrics_reject_burn_in_outside_evaluation_range(burn_in: int) -> None:
    with pytest.raises(ValueError):
        compute_run_level_market_outcomes(example_result(), burn_in=burn_in)


@pytest.mark.parametrize("burn_in", [True, 1.5, "1"])
def test_metrics_reject_noninteger_burn_in(burn_in) -> None:
    with pytest.raises(TypeError):
        compute_run_level_market_outcomes(example_result(), burn_in=burn_in)


def test_bundle_rejects_only_one_evaluated_period_because_rv_is_undefined() -> None:
    with pytest.raises(ValueError, match="at least two"):
        compute_run_level_market_outcomes(example_result(), burn_in=3)


def test_return_volatility_rejects_only_one_evaluated_period() -> None:
    with pytest.raises(ValueError, match="at least two"):
        return_volatility(example_result(), burn_in=3)


@pytest.mark.parametrize(
    "metric",
    [
        return_volatility,
        rms_mispricing,
        maximum_absolute_mispricing,
        mean_absolute_order_flow_per_agent,
        mean_absolute_return,
        time_averaged_belief_variance,
    ],
)
def test_public_metric_functions_require_simulation_result(metric) -> None:
    with pytest.raises(TypeError, match="SimulationResult"):
        metric("not-a-result")


def test_run_level_outcome_record_rejects_negative_metric() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RunLevelMarketOutcomes(
            burn_in=0,
            n_observations=4,
            return_volatility=-1.0,
            rms_mispricing=1.0,
            maximum_absolute_mispricing=1.0,
            mean_absolute_order_flow_per_agent=1.0,
            mean_absolute_return=1.0,
            time_averaged_belief_variance=1.0,
        )
