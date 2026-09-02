import numpy as np
import pytest

from src.experiments.refined import (
    CIDRunClassification,
    CIDThresholdConfiguration,
    OperationalStabilisationResult,
    RollingCIDPoint,
    classify_cid_path,
    operational_stabilisation,
    threshold_exceedance_rate,
)


def _point(
    endpoint: int,
    *,
    cid: float,
    ret: float = 1.0,
    bel: float = 1.0,
    flow: float = 1.0,
    window_length: int = 3,
) -> RollingCIDPoint:
    return RollingCIDPoint(
        endpoint_period=endpoint,
        window_start_period=endpoint - window_length + 1,
        window_length=window_length,
        rolling_return_volatility=ret,
        rolling_belief_dispersion=bel,
        rms_order_flow_pressure=flow,
        standardised_return=ret,
        standardised_belief=bel,
        standardised_order_flow=flow,
        cid=cid,
    )


def _path(cids, *, ret=None, bel=None, flow=None, first_endpoint=3):
    n = len(cids)
    ret = [1.0] * n if ret is None else ret
    bel = [1.0] * n if bel is None else bel
    flow = [1.0] * n if flow is None else flow
    return tuple(
        _point(
            first_endpoint + i,
            cid=cids[i],
            ret=ret[i],
            bel=bel[i],
            flow=flow[i],
        )
        for i in range(n)
    )


def test_inactive_guardrails_map_to_infinity():
    config = CIDThresholdConfiguration(cid_threshold=2.0)
    assert np.isinf(config.return_limit)
    assert np.isinf(config.belief_limit)
    assert np.isinf(config.order_flow_limit)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan, True])
def test_cid_threshold_must_be_positive_finite(bad):
    with pytest.raises((TypeError, ValueError)):
        CIDThresholdConfiguration(cid_threshold=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_return_volatility": 0.0},
        {"max_belief_dispersion": -1.0},
        {"max_order_flow_pressure": np.inf},
    ],
)
def test_active_guardrails_must_be_positive_finite(kwargs):
    with pytest.raises(ValueError):
        CIDThresholdConfiguration(cid_threshold=2.0, **kwargs)


def test_cid_crossing_classifies_run_as_threshold_exceeding():
    result = classify_cid_path(
        _path([1.0, 2.1, 1.0]),
        thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
        stabilisation_length=1,
    )
    assert result.threshold_exceeding is True


@pytest.mark.parametrize(
    "guardrail_name,series_name",
    [
        ("max_return_volatility", "ret"),
        ("max_belief_dispersion", "bel"),
        ("max_order_flow_pressure", "flow"),
    ],
)
def test_each_component_guardrail_can_trigger_run_exceedance(guardrail_name, series_name):
    series = [1.0, 3.0, 1.0]
    kwargs = {"ret": None, "bel": None, "flow": None}
    kwargs[series_name] = series
    points = _path([1.0, 1.0, 1.0], **kwargs)
    thresholds = CIDThresholdConfiguration(
        cid_threshold=2.0,
        **{guardrail_name: 2.0},
    )
    result = classify_cid_path(points, thresholds=thresholds, stabilisation_length=1)
    assert result.threshold_exceeding is True
    assert result.cid_exceedance_duration_share == 0.0


def test_exact_threshold_equality_is_not_an_exceedance_and_is_admissible():
    thresholds = CIDThresholdConfiguration(
        cid_threshold=2.0,
        max_return_volatility=1.0,
        max_belief_dispersion=1.0,
        max_order_flow_pressure=1.0,
    )
    result = classify_cid_path(
        _path([2.0, 2.0], ret=[1.0, 1.0], bel=[1.0, 1.0], flow=[1.0, 1.0]),
        thresholds=thresholds,
        stabilisation_length=2,
    )
    assert result.threshold_exceeding is False
    assert result.stabilisation.stabilised is True
    assert result.stabilisation.stabilisation_period == 3


def test_duration_share_counts_only_cid_crossings_not_guardrail_crossings():
    points = _path([1.0, 1.0, 1.0], ret=[1.0, 4.0, 1.0])
    result = classify_cid_path(
        points,
        thresholds=CIDThresholdConfiguration(
            cid_threshold=2.0,
            max_return_volatility=2.0,
        ),
        stabilisation_length=1,
    )
    assert result.threshold_exceeding is True
    assert result.cid_exceedance_duration_share == 0.0


def test_peak_and_duration_share_follow_equation_249():
    result = classify_cid_path(
        _path([1.0, 2.5, 3.0, 1.5]),
        thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
        stabilisation_length=1,
    )
    assert result.peak_cid == 3.0
    assert result.cid_exceedance_duration_share == 0.5


def test_stabilisation_returns_first_valid_consecutive_block():
    points = _path([3.0, 1.0, 1.0, 1.0, 1.0])
    result = operational_stabilisation(
        points,
        thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
        stabilisation_length=3,
    )
    assert result.stabilised is True
    assert result.right_censored is False
    assert result.stabilisation_period == 4
    assert result.last_eligible_start_period == 5


def test_interruption_delays_stabilisation_until_later_full_block():
    points = _path([1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
    result = operational_stabilisation(
        points,
        thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
        stabilisation_length=3,
    )
    assert result.stabilisation_period == 6


def test_guardrail_breach_prevents_otherwise_low_cid_block_from_stabilising():
    points = _path(
        [1.0, 1.0, 1.0, 1.0],
        ret=[3.0, 1.0, 1.0, 1.0],
    )
    result = operational_stabilisation(
        points,
        thresholds=CIDThresholdConfiguration(
            cid_threshold=2.0,
            max_return_volatility=2.0,
        ),
        stabilisation_length=2,
    )
    assert result.stabilisation_period == 4


def test_no_qualifying_block_is_right_censored_without_invented_time():
    result = operational_stabilisation(
        _path([3.0, 3.0, 3.0, 3.0]),
        thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
        stabilisation_length=2,
    )
    assert result.stabilised is False
    assert result.right_censored is True
    assert result.stabilisation_period is None
    assert result.last_eligible_start_period == 5


def test_stabilisation_length_longer_than_observed_path_is_right_censored():
    result = operational_stabilisation(
        _path([1.0, 1.0]),
        thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
        stabilisation_length=3,
    )
    assert result.right_censored is True
    assert result.stabilisation_period is None
    assert result.last_eligible_start_period is None


def test_default_first_stage_stabilisation_length_is_50():
    points = _path([1.0] * 50)
    result = operational_stabilisation(
        points,
        thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
    )
    assert result.stabilisation_length == 50
    assert result.stabilisation_period == 3


def test_threshold_exceedance_rate_is_mean_of_run_indicators():
    stab = OperationalStabilisationResult(
        stabilisation_length=1,
        stabilised=True,
        stabilisation_period=1,
        right_censored=False,
        last_eligible_start_period=1,
    )
    classifications = (
        CIDRunClassification(True, 3.0, 0.5, stab),
        CIDRunClassification(False, 1.0, 0.0, stab),
        CIDRunClassification(True, 4.0, 0.25, stab),
        CIDRunClassification(False, 1.5, 0.0, stab),
    )
    assert threshold_exceedance_rate(classifications) == 0.5


@pytest.mark.parametrize("bad", [[], (), "bad"])
def test_threshold_exceedance_rate_rejects_invalid_collections(bad):
    with pytest.raises((TypeError, ValueError)):
        threshold_exceedance_rate(bad)


def test_rolling_path_must_be_tuple():
    with pytest.raises(TypeError):
        classify_cid_path(
            list(_path([1.0, 1.0])),
            thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
            stabilisation_length=1,
        )


def test_rolling_path_must_be_nonempty():
    with pytest.raises(ValueError):
        classify_cid_path(
            (),
            thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
            stabilisation_length=1,
        )


def test_rolling_endpoints_must_be_consecutive():
    points = (_point(3, cid=1.0), _point(5, cid=1.0))
    with pytest.raises(ValueError, match="consecutive"):
        classify_cid_path(
            points,
            thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
            stabilisation_length=1,
        )


def test_rolling_points_must_share_window_length():
    points = (
        _point(3, cid=1.0, window_length=3),
        _point(4, cid=1.0, window_length=2),
    )
    with pytest.raises(ValueError, match="same window_length"):
        classify_cid_path(
            points,
            thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
            stabilisation_length=1,
        )


@pytest.mark.parametrize("bad", [0, -1, True])
def test_stabilisation_length_must_be_positive_integer(bad):
    with pytest.raises((TypeError, ValueError)):
        operational_stabilisation(
            _path([1.0, 1.0]),
            thresholds=CIDThresholdConfiguration(cid_threshold=2.0),
            stabilisation_length=bad,
        )
