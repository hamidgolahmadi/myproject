"""Canonical frozen numerical D042 market-evaluation calibration.

The values in this module come from the completed 500+500 no-social production
calibration under D042 and the frozen D043 market specification. They are
immutable inputs to the first confirmatory fixed-topology market experiments.

Do not retune these values after inspecting Random/Small-World/Hub treatment
outcomes. Any future recalibration requires an explicit new methodological
decision and new disjoint calibration seeds.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .baseline_specification import first_refined_baseline_specification
from .cid import CIDReferenceScales
from .market_calibration import (
    MarketEvaluationCalibration,
    first_market_evaluation_calibration_protocol,
)
from .market_calibration_run import calibration_configuration_fingerprint


FROZEN_C_RET = 0.0030364359162156455
FROZEN_C_BEL = 0.004182211355781272
FROZEN_C_F = 0.11381404220614316
FROZEN_C_CID = 1.8326578831721285

FROZEN_CONFIGURATION_FINGERPRINT = (
    "9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202"
)
FROZEN_REFERENCE_SCALES_FINGERPRINT = (
    "1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476"
)


def frozen_reference_scales() -> CIDReferenceScales:
    """Return the final D042 reference scales."""

    return CIDReferenceScales(
        return_scale=FROZEN_C_RET,
        belief_scale=FROZEN_C_BEL,
        order_flow_scale=FROZEN_C_F,
    )


def _reference_scales_fingerprint(scales: CIDReferenceScales) -> str:
    encoded = json.dumps(
        asdict(scales),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def first_frozen_market_evaluation_calibration() -> MarketEvaluationCalibration:
    """Return the canonical numerical D042 calibration with drift guards."""

    protocol = first_market_evaluation_calibration_protocol()
    baseline = first_refined_baseline_specification()

    configuration_fingerprint = calibration_configuration_fingerprint(
        protocol,
        baseline,
    )
    if configuration_fingerprint != FROZEN_CONFIGURATION_FINGERPRINT:
        raise RuntimeError(
            "current D042/D043 specification does not match the frozen calibration fingerprint"
        )

    scales = frozen_reference_scales()
    if _reference_scales_fingerprint(scales) != FROZEN_REFERENCE_SCALES_FINGERPRINT:
        raise RuntimeError("frozen D042 reference scales do not match their recorded fingerprint")

    return MarketEvaluationCalibration(
        protocol=protocol,
        reference_scales=scales,
        cid_weights=protocol.cid_weights,
        cid_threshold=FROZEN_C_CID,
    )
