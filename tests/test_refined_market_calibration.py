from dataclasses import replace

import numpy as np
import pytest

from src.experiments.refined.cid import (
    CIDReferenceScales,
    CIDWeights,
    RollingCIDComponentsPoint,
)
from src.experiments.refined.market_calibration import (
    MarketEvaluationCalibration,
    MarketEvaluationCalibrationProtocol,
    calibrate_market_evaluation,
    estimate_cid_threshold,
    estimate_reference_scales,
    first_market_evaluation_calibration_protocol,
)


def _protocol(**overrides) -> MarketEvaluationCalibrationProtocol:
    base = dict(
        scale_calibration_seed=101,
        threshold_calibration_seed=202,
        n_scale_replications=2,
        n_threshold_replications=3,
        horizon=5,
        burn_in=0,
        rolling_window=2,
        calibration_alpha=0.0,
        cid_weights=CIDWeights.equal(),
        cid_peak_quantile=0.5,
        stabilisation_length=2,
    )
    base.update(overrides)
    return MarketEvaluationCalibrationProtocol(**base)


def _point(endpoint: int, r: float, b: float, f: float, *, window: int = 2):
    return RollingCIDComponentsPoint(
        endpoint_period=endpoint,
        window_start_period=endpoint - window + 1,
        window_length=window,
        rolling_return_volatility=r,
        rolling_belief_dispersion=b,
        rms_order_flow_pressure=f,
    )


def _path(values, *, window=2):
    return tuple(
        _point(endpoint, r, b, f, window=window)
        for endpoint, (r, b, f) in enumerate(values, start=window)
    )


def test_first_protocol_freezes_explicit_baseline_design():
    protocol = first_market_evaluation_calibration_protocol()
    assert protocol.scale_calibration_seed == 2026090201
    assert protocol.threshold_calibration_seed == 2026090202
    assert protocol.scale_calibration_seed != protocol.threshold_calibration_seed
    assert protocol.n_scale_replications == 500
    assert protocol.n_threshold_replications == 500
    assert protocol.horizon == 1000
    assert protocol.burn_in == 0
    assert protocol.rolling_window == 50
    assert protocol.calibration_alpha == 0.0
    assert protocol.cid_weights == CIDWeights.equal()
    assert protocol.cid_peak_quantile == pytest.approx(0.95)
    assert protocol.stabilisation_length == 50
    assert protocol.robustness_windows == (25, 100)
    assert protocol.expected_rolling_points_per_run == 951


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("scale_calibration_seed", True, TypeError),
        ("scale_calibration_seed", -1, ValueError),
        ("n_scale_replications", 0, ValueError),
        ("n_threshold_replications", 0, ValueError),
        ("horizon", 0, ValueError),
        ("burn_in", -1, ValueError),
        ("rolling_window", 1, ValueError),
        ("stabilisation_length", 0, ValueError),
        ("calibration_alpha", 0.1, ValueError),
        ("calibration_alpha", True, TypeError),
        ("cid_peak_quantile", 0.0, ValueError),
        ("cid_peak_quantile", 1.0, ValueError),
    ],
)
def test_protocol_rejects_invalid_fields(field, value, error):
    kwargs = {
        "scale_calibration_seed": 101,
        "threshold_calibration_seed": 202,
        "n_scale_replications": 2,
        "n_threshold_replications": 3,
        "horizon": 5,
        "burn_in": 0,
        "rolling_window": 2,
        "calibration_alpha": 0.0,
        "cid_weights": CIDWeights.equal(),
        "cid_peak_quantile": 0.5,
        "stabilisation_length": 2,
    }
    kwargs[field] = value
    with pytest.raises(error):
        MarketEvaluationCalibrationProtocol(**kwargs)


def test_protocol_rejects_reused_scale_and_threshold_seed():
    with pytest.raises(ValueError, match="must be distinct"):
        _protocol(threshold_calibration_seed=101)


def test_protocol_rejects_burn_in_equal_to_horizon():
    with pytest.raises(ValueError, match="smaller than horizon"):
        _protocol(burn_in=5)


def test_protocol_rejects_window_longer_than_post_burn_sample():
    with pytest.raises(ValueError, match="cannot exceed"):
        _protocol(burn_in=4, rolling_window=2)


def test_protocol_rejects_non_weight_object():
    with pytest.raises(TypeError, match="CIDWeights"):
        _protocol(cid_weights=(1 / 3, 1 / 3, 1 / 3))


def test_expected_rolling_points_respects_burn_in():
    protocol = _protocol(horizon=10, burn_in=2, rolling_window=3)
    assert protocol.expected_rolling_points_per_run == 6


def test_reference_scales_are_pooled_component_medians():
    protocol = _protocol()
    scale_paths = (
        _path(((1, 2, 0.5), (2, 4, 1), (3, 6, 1.5), (4, 8, 2))),
        _path(((5, 10, 2.5), (6, 12, 3), (7, 14, 3.5), (8, 16, 4))),
    )
    scales = estimate_reference_scales(scale_paths, protocol=protocol)
    assert scales.return_scale == pytest.approx(4.5)
    assert scales.belief_scale == pytest.approx(9.0)
    assert scales.order_flow_scale == pytest.approx(2.25)


def test_reference_scale_calibration_rejects_zero_median_instead_of_epsilon_patch():
    protocol = _protocol()
    scale_paths = (
        _path(((0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1))),
        _path(((0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1))),
    )
    with pytest.raises(ValueError, match="non-positive reference scale"):
        estimate_reference_scales(scale_paths, protocol=protocol)


def test_reference_scale_calibration_requires_exact_replication_count():
    protocol = _protocol()
    one_path = _path(((1, 1, 1),) * 4)
    with pytest.raises(ValueError, match="exactly 2 replications"):
        estimate_reference_scales((one_path,), protocol=protocol)


def test_calibration_requires_tuple_paths():
    protocol = _protocol()
    path = _path(((1, 1, 1),) * 4)
    with pytest.raises(TypeError, match="paths must be tuples"):
        estimate_reference_scales((list(path), tuple(path)), protocol=protocol)


def test_calibration_requires_exact_rolling_point_count():
    protocol = _protocol()
    short = _path(((1, 1, 1),) * 3)
    full = _path(((1, 1, 1),) * 4)
    with pytest.raises(ValueError, match="exactly 4 rolling points"):
        estimate_reference_scales((short, full), protocol=protocol)


def test_calibration_rejects_wrong_window_length():
    protocol = _protocol()
    wrong = _path(((1, 1, 1),) * 4, window=3)
    with pytest.raises(ValueError, match="unexpected rolling-window"):
        estimate_reference_scales((wrong, wrong), protocol=protocol)


def test_calibration_rejects_non_consecutive_endpoints():
    protocol = _protocol()
    path = list(_path(((1, 1, 1),) * 4))
    path[2] = _point(5, 1, 1, 1)
    with pytest.raises(ValueError, match="must be consecutive"):
        estimate_reference_scales((tuple(path), _path(((1, 1, 1),) * 4)), protocol=protocol)


def test_threshold_uses_run_level_peak_cid():
    protocol = _protocol(cid_peak_quantile=0.5)
    scales = CIDReferenceScales(1.0, 1.0, 1.0)
    threshold_paths = (
        _path(((0.1, 0.1, 0.1), (1.0, 1.0, 1.0), (0.2, 0.2, 0.2), (0.1, 0.1, 0.1))),
        _path(((0.1, 0.1, 0.1), (3.0, 3.0, 3.0), (0.2, 0.2, 0.2), (0.1, 0.1, 0.1))),
        _path(((0.1, 0.1, 0.1), (2.0, 2.0, 2.0), (0.2, 0.2, 0.2), (0.1, 0.1, 0.1))),
    )
    # Run-level peaks are [1, 3, 2]; the median with method='higher' is 2.
    assert estimate_cid_threshold(
        threshold_paths,
        protocol=protocol,
        scales=scales,
    ) == pytest.approx(2.0)


def test_threshold_higher_quantile_is_conservative_in_finite_sample():
    protocol = _protocol(cid_peak_quantile=0.75, n_threshold_replications=4)
    scales = CIDReferenceScales(1.0, 1.0, 1.0)
    paths = tuple(
        _path(((peak, peak, peak),) * 4)
        for peak in (1.0, 2.0, 3.0, 4.0)
    )
    assert estimate_cid_threshold(paths, protocol=protocol, scales=scales) == pytest.approx(4.0)


def test_threshold_sample_count_is_separate_from_scale_sample_count():
    protocol = _protocol(n_scale_replications=1, n_threshold_replications=3)
    scales = CIDReferenceScales(1.0, 1.0, 1.0)
    path = _path(((1, 1, 1),) * 4)
    assert estimate_cid_threshold(
        (path, path, path),
        protocol=protocol,
        scales=scales,
    ) == pytest.approx(1.0)


def test_threshold_requires_reference_scale_object():
    protocol = _protocol()
    path = _path(((1, 1, 1),) * 4)
    with pytest.raises(TypeError, match="CIDReferenceScales"):
        estimate_cid_threshold((path, path, path), protocol=protocol, scales=(1, 1, 1))


def test_full_calibration_returns_scales_threshold_weights_and_no_baseline_guardrails():
    protocol = _protocol()
    scale_paths = (
        _path(((1, 2, 3),) * 4),
        _path(((3, 4, 5),) * 4),
    )
    threshold_paths = (
        _path(((2, 3, 4),) * 4),
        _path(((4, 6, 8),) * 4),
        _path(((3, 4.5, 6),) * 4),
    )
    calibration = calibrate_market_evaluation(
        scale_paths,
        threshold_paths,
        protocol=protocol,
    )
    assert isinstance(calibration, MarketEvaluationCalibration)
    assert calibration.reference_scales == CIDReferenceScales(2.0, 3.0, 4.0)
    assert calibration.cid_weights == CIDWeights.equal()
    assert calibration.cid_threshold == pytest.approx(1.5)
    assert calibration.return_guardrail is None
    assert calibration.belief_guardrail is None
    assert calibration.order_flow_guardrail is None


def test_market_calibration_rejects_weight_mismatch_with_protocol():
    protocol = _protocol()
    with pytest.raises(ValueError, match="must match"):
        MarketEvaluationCalibration(
            protocol=protocol,
            reference_scales=CIDReferenceScales(1.0, 1.0, 1.0),
            cid_weights=CIDWeights(1.0, 0.0, 0.0),
            cid_threshold=2.0,
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.inf, np.nan])
def test_market_calibration_rejects_invalid_threshold(threshold):
    protocol = _protocol()
    with pytest.raises(ValueError, match="strictly positive"):
        MarketEvaluationCalibration(
            protocol=protocol,
            reference_scales=CIDReferenceScales(1.0, 1.0, 1.0),
            cid_weights=protocol.cid_weights,
            cid_threshold=threshold,
        )


def test_protocol_is_immutable_and_replace_can_create_explicit_robustness_variant():
    protocol = first_market_evaluation_calibration_protocol()
    robustness = replace(protocol, rolling_window=25)
    assert protocol.rolling_window == 50
    assert robustness.rolling_window == 25
    assert robustness.expected_rolling_points_per_run == 976
