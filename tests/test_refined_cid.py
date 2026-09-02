import numpy as np
import pytest

from src.experiments.refined import (
    CIDReferenceScales,
    CIDWeights,
    RollingCIDComponentsPoint,
    RollingCIDPoint,
    rolling_cid,
    rolling_cid_components,
    standardise_cid_components,
)
from src.model.refined import PeriodOutputs, RefinedState, SimulationResult


def _state(beliefs):
    beliefs = np.asarray(beliefs, dtype=float)
    n = beliefs.size
    if n == 1:
        attention = np.ones((1, 1), dtype=float)
    else:
        attention = np.zeros((n, n), dtype=float)
        for i in range(n):
            attention[i, (i + 1) % n] = 1.0
    return RefinedState(
        theta=0.0,
        beliefs=beliefs,
        positions=np.zeros(n),
        price=0.0,
        reputation=np.zeros(n),
        attention=attention,
    )


def _output(n, *, return_, flow, actions=None):
    if actions is None:
        actions = np.full(n, flow / n, dtype=float)
    actions = np.asarray(actions, dtype=float)
    zeros = np.zeros(n, dtype=float)
    return PeriodOutputs(
        fundamental_value=0.0,
        signals=zeros,
        perceived_values=zeros,
        valuation_gaps=zeros,
        desired_actions=actions,
        actions=actions,
        net_order_flow=flow,
        return_=return_,
        profits=zeros,
        reputation_scores=np.zeros((n, n), dtype=float),
    )


def _result():
    # Cross-sectional population variances at t=1,...,5 are 1,4,9,16,25.
    states = (
        _state([0.0, 0.0]),
        _state([0.0, 2.0]),
        _state([1.0, 5.0]),
        _state([2.0, 8.0]),
        _state([3.0, 11.0]),
        _state([4.0, 14.0]),
    )
    returns = [1.0, 2.0, 4.0, 8.0, 16.0]
    flows = [2.0, 0.0, -2.0, 4.0, -4.0]
    outputs = tuple(
        _output(2, return_=r, flow=f)
        for r, f in zip(returns, flows, strict=True)
    )
    return SimulationResult(states=states, period_outputs=outputs)


def test_first_window_components_are_exact():
    point = rolling_cid_components(_result(), window_length=3)[0]

    assert point.endpoint_period == 3
    assert point.window_start_period == 1
    assert point.window_length == 3
    assert point.rolling_return_volatility == pytest.approx(np.sqrt(7.0 / 3.0))
    assert point.rolling_belief_dispersion == pytest.approx(14.0 / 3.0)
    assert point.rms_order_flow_pressure == pytest.approx(np.sqrt(2.0 / 3.0))


def test_all_valid_endpoints_are_returned():
    points = rolling_cid_components(_result(), window_length=3)
    assert [p.endpoint_period for p in points] == [3, 4, 5]
    assert [p.window_start_period for p in points] == [1, 2, 3]


def test_burn_in_changes_first_endpoint_and_window():
    points = rolling_cid_components(_result(), window_length=2, burn_in=1)
    assert [p.endpoint_period for p in points] == [3, 4, 5]
    assert [p.window_start_period for p in points] == [2, 3, 4]


def test_return_component_uses_sample_standard_deviation():
    point = rolling_cid_components(_result(), window_length=3)[0]
    sample_sd = np.std([1.0, 2.0, 4.0], ddof=1)
    population_sd = np.std([1.0, 2.0, 4.0], ddof=0)
    assert point.rolling_return_volatility == pytest.approx(sample_sd)
    assert point.rolling_return_volatility != pytest.approx(population_sd)


def test_belief_component_uses_population_cross_sectional_variance():
    point = rolling_cid_components(_result(), window_length=3)[0]
    expected_variances = [1.0, 4.0, 9.0]
    assert point.rolling_belief_dispersion == pytest.approx(np.mean(expected_variances))
    # It is not the average cross-sectional standard deviation.
    assert point.rolling_belief_dispersion != pytest.approx(np.mean([1.0, 2.0, 3.0]))


def test_order_flow_component_uses_signed_net_flow_per_agent():
    point = rolling_cid_components(_result(), window_length=3)[0]
    expected = np.sqrt(np.mean(np.square(np.array([2.0, 0.0, -2.0]) / 2.0)))
    assert point.rms_order_flow_pressure == pytest.approx(expected)


def test_order_flow_component_does_not_use_gross_action_volume():
    states = tuple(_state([0.0, 0.0]) for _ in range(4))
    outputs = (
        _output(2, return_=0.0, flow=0.0, actions=[1.0, -1.0]),
        _output(2, return_=0.0, flow=0.0, actions=[0.5, -0.5]),
        _output(2, return_=0.0, flow=0.0, actions=[1.0, -1.0]),
    )
    result = SimulationResult(states=states, period_outputs=outputs)
    point = rolling_cid_components(result, window_length=3)[0]
    assert point.rms_order_flow_pressure == 0.0


def test_one_agent_is_supported():
    states = (
        _state([0.0]),
        _state([1.0]),
        _state([2.0]),
    )
    outputs = (
        _output(1, return_=1.0, flow=2.0),
        _output(1, return_=3.0, flow=-2.0),
    )
    result = SimulationResult(states=states, period_outputs=outputs)
    point = rolling_cid_components(result, window_length=2)[0]
    assert point.rolling_belief_dispersion == 0.0
    assert point.rms_order_flow_pressure == 2.0


def test_standardisation_is_exact():
    components = (
        RollingCIDComponentsPoint(
            endpoint_period=3,
            window_start_period=1,
            window_length=3,
            rolling_return_volatility=2.0,
            rolling_belief_dispersion=8.0,
            rms_order_flow_pressure=0.5,
        ),
    )
    scales = CIDReferenceScales(1.0, 4.0, 0.25)
    weights = CIDWeights(0.2, 0.3, 0.5)
    point = standardise_cid_components(components, scales=scales, weights=weights)[0]

    assert point.standardised_return == 2.0
    assert point.standardised_belief == 2.0
    assert point.standardised_order_flow == 2.0
    assert point.cid == 2.0


def test_weighted_cid_uses_equation_246():
    components = (
        RollingCIDComponentsPoint(3, 1, 3, 3.0, 4.0, 5.0),
    )
    scales = CIDReferenceScales(1.0, 2.0, 5.0)
    weights = CIDWeights(0.5, 0.25, 0.25)
    point = standardise_cid_components(components, scales=scales, weights=weights)[0]
    expected = 0.5 * 3.0 + 0.25 * 2.0 + 0.25 * 1.0
    assert point.cid == pytest.approx(expected)


def test_equal_weight_factory():
    weights = CIDWeights.equal()
    assert weights.return_weight == pytest.approx(1.0 / 3.0)
    assert weights.belief_weight == pytest.approx(1.0 / 3.0)
    assert weights.order_flow_weight == pytest.approx(1.0 / 3.0)


def test_rolling_cid_matches_two_stage_computation():
    result = _result()
    scales = CIDReferenceScales(2.0, 5.0, 0.75)
    weights = CIDWeights.equal()
    direct = rolling_cid(
        result,
        window_length=3,
        scales=scales,
        weights=weights,
    )
    staged = standardise_cid_components(
        rolling_cid_components(result, window_length=3),
        scales=scales,
        weights=weights,
    )
    assert direct == staged


def test_component_point_rejects_inconsistent_window_metadata():
    with pytest.raises(ValueError):
        RollingCIDComponentsPoint(4, 1, 2, 1.0, 1.0, 1.0)


def test_cid_point_rejects_negative_cid():
    with pytest.raises(ValueError):
        RollingCIDPoint(2, 1, 2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.1)


@pytest.mark.parametrize(
    ("window_length", "burn_in", "error"),
    [
        (0, 0, ValueError),
        (1, 0, ValueError),
        (6, 0, ValueError),
        (True, 0, TypeError),
        (3, -1, ValueError),
        (3, 5, ValueError),
        (3, 1.5, TypeError),
    ],
)
def test_invalid_rolling_inputs_are_rejected(window_length, burn_in, error):
    with pytest.raises(error):
        rolling_cid_components(
            _result(),
            window_length=window_length,
            burn_in=burn_in,
        )


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, np.nan),
        (True, 1.0, 1.0),
        (1.0, np.inf, 1.0),
        (1.0, 1.0, "bad"),
    ],
)
def test_invalid_reference_scales_are_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        CIDReferenceScales(*values)


@pytest.mark.parametrize(
    "values",
    [
        (-0.1, 0.6, 0.5),
        (0.3, 0.3, 0.3),
        (np.nan, 0.5, 0.5),
        (True, 0.0, 0.0),
        (1.1, 0.0, -0.1),
        (0.0, 0.0, 0.0),
    ],
)
def test_invalid_cid_weights_are_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        CIDWeights(*values)


def test_standardisation_requires_scale_object():
    components = (RollingCIDComponentsPoint(2, 1, 2, 1.0, 1.0, 1.0),)
    with pytest.raises(TypeError):
        standardise_cid_components(
            components,
            scales=(1.0, 1.0, 1.0),
            weights=CIDWeights.equal(),
        )


def test_standardisation_requires_weight_object():
    components = (RollingCIDComponentsPoint(2, 1, 2, 1.0, 1.0, 1.0),)
    with pytest.raises(TypeError):
        standardise_cid_components(
            components,
            scales=CIDReferenceScales(1.0, 1.0, 1.0),
            weights=(1.0 / 3.0,) * 3,
        )


def test_standardisation_rejects_empty_components():
    with pytest.raises(ValueError):
        standardise_cid_components(
            (),
            scales=CIDReferenceScales(1.0, 1.0, 1.0),
            weights=CIDWeights.equal(),
        )


def test_standardisation_rejects_non_point_entries():
    with pytest.raises(TypeError):
        standardise_cid_components(
            ("bad",),
            scales=CIDReferenceScales(1.0, 1.0, 1.0),
            weights=CIDWeights.equal(),
        )


def test_wrong_result_type_is_rejected():
    with pytest.raises(TypeError):
        rolling_cid_components("not a simulation", window_length=2)


def test_inconsistent_state_agent_dimensions_are_rejected():
    result = SimulationResult(
        states=(
            _state([0.0, 0.0]),
            _state([0.0, 0.0]),
            _state([0.0, 0.0, 0.0]),
        ),
        period_outputs=(
            _output(2, return_=0.0, flow=0.0),
            _output(2, return_=0.0, flow=0.0),
        ),
    )
    with pytest.raises(ValueError):
        rolling_cid_components(result, window_length=2)
