"""End-to-end no-social smoke for the frozen D042 calibration method.

This module validates the complete calibration pipeline under the frozen D043
market specification before the full 500+500 calibration is submitted.  Smoke
seeds are deliberately disjoint from the final D042 scale/threshold namespaces,
because smoke outputs are inspected during development and therefore must never
be recycled into the final calibration sample.

The smoke uses one canonical directed Random fixed-out-degree graph per
replication.  At alpha=0 the graph and attention process do not enter beliefs or
market outcomes; the graph exists only to keep the full state/attention
machinery valid.  No topology comparison or ranking is produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from src.model.refined import (
    generate_shock_path,
    initialise_state,
    simulate_shock_path,
    uniform_attention_from_graph,
)
from src.topologies.refined import generate_random_fixed_out_degree

from .baseline_specification import (
    RefinedBaselineSpecification,
    first_refined_baseline_specification,
    generate_neutral_nonnetwork_initial_conditions,
)
from .cid import RollingCIDComponentsPoint, rolling_cid_components, standardise_cid_components
from .market_calibration import (
    MarketEvaluationCalibration,
    MarketEvaluationCalibrationProtocol,
    calibrate_market_evaluation,
    first_market_evaluation_calibration_protocol,
)
from .seeding import derive_graph_seed, derive_semantic_seed, nonnegative_integer


_SMOKE_PURPOSE = "small no-social end-to-end calibration smoke; not final D042 calibration"
_CALIBRATION_GRAPH_LABEL = "CAL"


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


@dataclass(frozen=True, slots=True)
class NoSocialCalibrationSmokeProtocol:
    """Small seed-disjoint version of the frozen D042 calibration design."""

    scale_seed: int = 2026090204
    threshold_seed: int = 2026090205
    n_scale_replications: int = 3
    n_threshold_replications: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_seed", nonnegative_integer("scale_seed", self.scale_seed))
        object.__setattr__(
            self,
            "threshold_seed",
            nonnegative_integer("threshold_seed", self.threshold_seed),
        )
        if self.scale_seed == self.threshold_seed:
            raise ValueError("scale_seed and threshold_seed must be distinct")
        object.__setattr__(
            self,
            "n_scale_replications",
            _positive_integer("n_scale_replications", self.n_scale_replications),
        )
        object.__setattr__(
            self,
            "n_threshold_replications",
            _positive_integer("n_threshold_replications", self.n_threshold_replications),
        )

        final_protocol = first_market_evaluation_calibration_protocol()
        forbidden = {
            final_protocol.scale_calibration_seed,
            final_protocol.threshold_calibration_seed,
        }
        if self.scale_seed in forbidden or self.threshold_seed in forbidden:
            raise ValueError("smoke seeds must be disjoint from final D042 calibration seeds")

    def market_protocol(self) -> MarketEvaluationCalibrationProtocol:
        """Return D042 methodology with only smoke counts/namespaces substituted."""

        final = first_market_evaluation_calibration_protocol()
        return MarketEvaluationCalibrationProtocol(
            scale_calibration_seed=self.scale_seed,
            threshold_calibration_seed=self.threshold_seed,
            n_scale_replications=self.n_scale_replications,
            n_threshold_replications=self.n_threshold_replications,
            horizon=final.horizon,
            burn_in=final.burn_in,
            rolling_window=final.rolling_window,
            calibration_alpha=final.calibration_alpha,
            cid_weights=final.cid_weights,
            cid_peak_quantile=final.cid_peak_quantile,
            stabilisation_length=final.stabilisation_length,
        )


@dataclass(frozen=True, slots=True)
class NoSocialCalibrationSmokeResult:
    smoke_protocol: NoSocialCalibrationSmokeProtocol
    market_protocol: MarketEvaluationCalibrationProtocol
    baseline: RefinedBaselineSpecification
    calibration: MarketEvaluationCalibration
    scale_paths: tuple[tuple[RollingCIDComponentsPoint, ...], ...]
    threshold_paths: tuple[tuple[RollingCIDComponentsPoint, ...], ...]
    threshold_peak_cids: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.smoke_protocol, NoSocialCalibrationSmokeProtocol):
            raise TypeError("smoke_protocol must be NoSocialCalibrationSmokeProtocol")
        if not isinstance(self.market_protocol, MarketEvaluationCalibrationProtocol):
            raise TypeError("market_protocol must be MarketEvaluationCalibrationProtocol")
        if not isinstance(self.baseline, RefinedBaselineSpecification):
            raise TypeError("baseline must be RefinedBaselineSpecification")
        if not isinstance(self.calibration, MarketEvaluationCalibration):
            raise TypeError("calibration must be MarketEvaluationCalibration")
        if len(self.scale_paths) != self.smoke_protocol.n_scale_replications:
            raise ValueError("unexpected number of scale paths")
        if len(self.threshold_paths) != self.smoke_protocol.n_threshold_replications:
            raise ValueError("unexpected number of threshold paths")
        if len(self.threshold_peak_cids) != self.smoke_protocol.n_threshold_replications:
            raise ValueError("unexpected number of threshold peak CID values")
        if not all(np.isfinite(value) and value > 0.0 for value in self.threshold_peak_cids):
            raise ValueError("threshold peak CID values must be finite and positive")



def _no_social_parameters(baseline: RefinedBaselineSpecification):
    parameters = replace(baseline.parameters, alpha=0.0)
    if parameters.alpha != 0.0:
        raise RuntimeError("calibration smoke failed to impose alpha=0")
    return parameters


def _component_path(
    *,
    experiment_seed: int,
    replication_id: int,
    baseline: RefinedBaselineSpecification,
    protocol: MarketEvaluationCalibrationProtocol,
) -> tuple[RollingCIDComponentsPoint, ...]:
    """Generate one canonical no-social calibration component path."""

    parameters = _no_social_parameters(baseline)
    shock_seed = derive_semantic_seed(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        role="shock",
    )
    initial_state_seed = derive_semantic_seed(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        role="initial_state",
    )
    graph_seed = derive_graph_seed(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        topology_label=_CALIBRATION_GRAPH_LABEL,
    )

    graph = generate_random_fixed_out_degree(
        n_agents=baseline.n_agents,
        k=baseline.k,
        graph_seed=graph_seed,
    )
    attention = uniform_attention_from_graph(graph)
    initial = generate_neutral_nonnetwork_initial_conditions(
        n_agents=baseline.n_agents,
        parameters=parameters,
        initial_state_seed=initial_state_seed,
    )
    initial_state = initialise_state(
        theta=initial.theta,
        beliefs=initial.beliefs,
        positions=initial.positions,
        price=initial.price,
        reputation=initial.reputation,
        attention=attention,
        graph=graph,
        x_bar=parameters.x_bar,
    )
    shock_path = generate_shock_path(
        n_periods=protocol.horizon,
        n_agents=baseline.n_agents,
        parameters=parameters,
        shock_seed=shock_seed,
    )
    simulation = simulate_shock_path(
        initial_state,
        shock_path,
        graph,
        parameters,
        adaptive_attention=True,
    )
    components = rolling_cid_components(
        simulation,
        window_length=protocol.rolling_window,
        burn_in=protocol.burn_in,
    )
    if len(components) != protocol.expected_rolling_points_per_run:
        raise RuntimeError("unexpected number of rolling calibration endpoints")
    return components


def run_no_social_calibration_smoke(
    *,
    smoke_protocol: NoSocialCalibrationSmokeProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
) -> NoSocialCalibrationSmokeResult:
    """Run the complete seed-disjoint smoke version of D042."""

    if smoke_protocol is None:
        smoke_protocol = NoSocialCalibrationSmokeProtocol()
    if baseline is None:
        baseline = first_refined_baseline_specification()
    if not isinstance(smoke_protocol, NoSocialCalibrationSmokeProtocol):
        raise TypeError("smoke_protocol must be NoSocialCalibrationSmokeProtocol")
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")

    protocol = smoke_protocol.market_protocol()
    if baseline.horizon != protocol.horizon:
        raise ValueError("baseline horizon must match the calibration protocol horizon")

    scale_paths = tuple(
        _component_path(
            experiment_seed=smoke_protocol.scale_seed,
            replication_id=replication_id,
            baseline=baseline,
            protocol=protocol,
        )
        for replication_id in range(smoke_protocol.n_scale_replications)
    )
    threshold_paths = tuple(
        _component_path(
            experiment_seed=smoke_protocol.threshold_seed,
            replication_id=replication_id,
            baseline=baseline,
            protocol=protocol,
        )
        for replication_id in range(smoke_protocol.n_threshold_replications)
    )

    calibration = calibrate_market_evaluation(
        scale_paths,
        threshold_paths,
        protocol=protocol,
    )
    peak_cids = []
    for path in threshold_paths:
        cid_path = standardise_cid_components(
            path,
            scales=calibration.reference_scales,
            weights=calibration.cid_weights,
        )
        peak_cids.append(max(point.cid for point in cid_path))

    return NoSocialCalibrationSmokeResult(
        smoke_protocol=smoke_protocol,
        market_protocol=protocol,
        baseline=baseline,
        calibration=calibration,
        scale_paths=scale_paths,
        threshold_paths=threshold_paths,
        threshold_peak_cids=tuple(float(value) for value in peak_cids),
    )


def write_no_social_calibration_smoke(
    result: NoSocialCalibrationSmokeResult,
    *,
    outdir: str | Path,
) -> Path:
    """Persist a clearly-labelled smoke-only calibration artifact as JSON."""

    if not isinstance(result, NoSocialCalibrationSmokeResult):
        raise TypeError("result must be NoSocialCalibrationSmokeResult")
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration_smoke.json"

    scales = result.calibration.reference_scales
    weights = result.calibration.cid_weights
    payload = {
        "purpose": _SMOKE_PURPOSE,
        "final_calibration": False,
        "calibration_alpha": result.market_protocol.calibration_alpha,
        "expected_rolling_points_per_run": result.market_protocol.expected_rolling_points_per_run,
        "smoke_protocol": asdict(result.smoke_protocol),
        "method": {
            "horizon": result.market_protocol.horizon,
            "burn_in": result.market_protocol.burn_in,
            "rolling_window": result.market_protocol.rolling_window,
            "cid_peak_quantile": result.market_protocol.cid_peak_quantile,
            "stabilisation_length": result.market_protocol.stabilisation_length,
        },
        "reference_scales": {
            "c_ret": scales.return_scale,
            "c_bel": scales.belief_scale,
            "c_F": scales.order_flow_scale,
        },
        "cid_weights": {
            "return": weights.return_weight,
            "belief": weights.belief_weight,
            "order_flow": weights.order_flow_weight,
        },
        "c_CID": result.calibration.cid_threshold,
        "threshold_peak_cids": list(result.threshold_peak_cids),
        "frozen_baseline_sigma_0": result.baseline.parameters.sigma_0,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
