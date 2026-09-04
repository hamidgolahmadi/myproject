import pytest

from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.frozen_market_calibration import (
    FROZEN_C_BEL,
    FROZEN_C_CID,
    FROZEN_C_F,
    FROZEN_C_RET,
    FROZEN_CONFIGURATION_FINGERPRINT,
    FROZEN_REFERENCE_SCALES_FINGERPRINT,
    _reference_scales_fingerprint,
    first_frozen_market_evaluation_calibration,
    frozen_reference_scales,
)
from src.experiments.refined.market_calibration import first_market_evaluation_calibration_protocol
from src.experiments.refined.market_calibration_run import calibration_configuration_fingerprint


def test_frozen_calibration_exact_numerical_values():
    assert FROZEN_C_RET == 0.0030364359162156455
    assert FROZEN_C_BEL == 0.004182211355781272
    assert FROZEN_C_F == 0.11381404220614316
    assert FROZEN_C_CID == 1.8326578831721285


def test_frozen_configuration_fingerprint_matches_current_d042_d043():
    actual = calibration_configuration_fingerprint(
        first_market_evaluation_calibration_protocol(),
        first_refined_baseline_specification(),
    )
    assert actual == FROZEN_CONFIGURATION_FINGERPRINT


def test_frozen_reference_scale_fingerprint_matches_values():
    assert _reference_scales_fingerprint(frozen_reference_scales()) == (
        FROZEN_REFERENCE_SCALES_FINGERPRINT
    )


def test_frozen_calibration_uses_final_500_plus_500_protocol():
    calibration = first_frozen_market_evaluation_calibration()
    protocol = calibration.protocol
    assert protocol.n_scale_replications == 500
    assert protocol.n_threshold_replications == 500
    assert protocol.scale_calibration_seed == 2026090201
    assert protocol.threshold_calibration_seed == 2026090202
    assert protocol.calibration_alpha == 0.0


def test_frozen_calibration_uses_equal_weights_and_no_guardrails():
    calibration = first_frozen_market_evaluation_calibration()
    assert calibration.cid_weights.return_weight == pytest.approx(1.0 / 3.0)
    assert calibration.cid_weights.belief_weight == pytest.approx(1.0 / 3.0)
    assert calibration.cid_weights.order_flow_weight == pytest.approx(1.0 / 3.0)
    assert calibration.return_guardrail is None
    assert calibration.belief_guardrail is None
    assert calibration.order_flow_guardrail is None


def test_frozen_calibration_object_matches_exact_scales_and_threshold():
    calibration = first_frozen_market_evaluation_calibration()
    assert calibration.reference_scales.return_scale == FROZEN_C_RET
    assert calibration.reference_scales.belief_scale == FROZEN_C_BEL
    assert calibration.reference_scales.order_flow_scale == FROZEN_C_F
    assert calibration.cid_threshold == FROZEN_C_CID
