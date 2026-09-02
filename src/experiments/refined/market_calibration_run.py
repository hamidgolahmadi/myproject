"""Resumable production runner for the frozen D042 market calibration.

The final calibration consists of two independent no-social samples under the
frozen D043 baseline:

1. 500 scale replications under namespace 2026090201;
2. 500 threshold replications under namespace 2026090202.

Each replication is checkpointed independently so an interrupted batch job can
resume without regenerating completed paths.  At alpha=0, adaptive attention is
an exact computational irrelevance for returns, beliefs, order flow, and the
CID components.  Production calibration therefore carries fixed graph-supported
attention forward (``adaptive_attention=False``); exact equivalence to the fully
adaptive path is regression-tested under common graph/shock/initial-state
randomness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .baseline_specification import (
    RefinedBaselineSpecification,
    first_refined_baseline_specification,
)
from .cid import CIDReferenceScales, standardise_cid_components
from .market_calibration import (
    MarketEvaluationCalibration,
    MarketEvaluationCalibrationProtocol,
    first_market_evaluation_calibration_protocol,
)
from .no_social_calibration_paths import no_social_component_path


_FINAL_ARTIFACT_NAME = "market_evaluation_calibration.json"
_SCALE_ARTIFACT_NAME = "reference_scales.json"
_THRESHOLD_CSV_NAME = "threshold_peak_cids.csv"
_QUANTILE_METHOD = "higher"
ProgressCallback = Callable[[str, int, int, bool], None]


@dataclass(frozen=True, slots=True)
class FinalMarketCalibrationRun:
    """Completed numerical D042 calibration and audit information."""

    protocol: MarketEvaluationCalibrationProtocol
    baseline: RefinedBaselineSpecification
    calibration: MarketEvaluationCalibration
    threshold_peak_cids: tuple[float, ...]
    output_dir: Path

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, MarketEvaluationCalibrationProtocol):
            raise TypeError("protocol must be MarketEvaluationCalibrationProtocol")
        if not isinstance(self.baseline, RefinedBaselineSpecification):
            raise TypeError("baseline must be RefinedBaselineSpecification")
        if not isinstance(self.calibration, MarketEvaluationCalibration):
            raise TypeError("calibration must be MarketEvaluationCalibration")
        if len(self.threshold_peak_cids) != self.protocol.n_threshold_replications:
            raise ValueError("unexpected number of threshold peak CID values")
        if not all(np.isfinite(value) and value > 0.0 for value in self.threshold_peak_cids):
            raise ValueError("threshold peak CID values must be finite and positive")
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _scale_checkpoint_path(output_dir: Path, replication_id: int) -> Path:
    return output_dir / "scale" / f"replication_{replication_id:04d}.npz"


def _threshold_checkpoint_path(output_dir: Path, replication_id: int) -> Path:
    return output_dir / "threshold" / f"replication_{replication_id:04d}.json"


def _component_arrays(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([point.rolling_return_volatility for point in path], dtype=float),
        np.asarray([point.rolling_belief_dispersion for point in path], dtype=float),
        np.asarray([point.rms_order_flow_pressure for point in path], dtype=float),
    )


def _write_scale_checkpoint(
    checkpoint: Path,
    *,
    replication_id: int,
    experiment_seed: int,
    return_values: np.ndarray,
    belief_values: np.ndarray,
    flow_values: np.ndarray,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            replication_id=np.asarray(replication_id, dtype=np.int64),
            experiment_seed=np.asarray(experiment_seed, dtype=np.uint64),
            return_values=return_values,
            belief_values=belief_values,
            flow_values=flow_values,
        )
    os.replace(temporary, checkpoint)


def _read_scale_checkpoint(
    checkpoint: Path,
    *,
    replication_id: int,
    experiment_seed: int,
    expected_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(checkpoint, allow_pickle=False) as payload:
        stored_replication = int(payload["replication_id"])
        stored_seed = int(payload["experiment_seed"])
        arrays = (
            np.asarray(payload["return_values"], dtype=float),
            np.asarray(payload["belief_values"], dtype=float),
            np.asarray(payload["flow_values"], dtype=float),
        )
    if stored_replication != replication_id or stored_seed != experiment_seed:
        raise ValueError(f"scale checkpoint metadata mismatch: {checkpoint}")
    if any(array.shape != (expected_points,) for array in arrays):
        raise ValueError(f"scale checkpoint has unexpected shape: {checkpoint}")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"scale checkpoint contains non-finite values: {checkpoint}")
    return arrays


def _validate_inputs(
    protocol: MarketEvaluationCalibrationProtocol,
    baseline: RefinedBaselineSpecification,
) -> None:
    if not isinstance(protocol, MarketEvaluationCalibrationProtocol):
        raise TypeError("protocol must be MarketEvaluationCalibrationProtocol")
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if protocol.calibration_alpha != 0.0:
        raise ValueError("production calibration requires alpha=0")
    if baseline.horizon != protocol.horizon:
        raise ValueError("baseline horizon must match calibration protocol horizon")


def run_scale_calibration_stage(
    *,
    output_dir: str | Path,
    protocol: MarketEvaluationCalibrationProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> CIDReferenceScales:
    """Run/resume the independent D042 scale sample and return pooled medians."""

    protocol = protocol or first_market_evaluation_calibration_protocol()
    baseline = baseline or first_refined_baseline_specification()
    _validate_inputs(protocol, baseline)
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")

    output_dir = Path(output_dir)
    expected_points = protocol.expected_rolling_points_per_run
    all_return: list[np.ndarray] = []
    all_belief: list[np.ndarray] = []
    all_flow: list[np.ndarray] = []

    for replication_id in range(protocol.n_scale_replications):
        checkpoint = _scale_checkpoint_path(output_dir, replication_id)
        reused = checkpoint.exists() and resume
        if reused:
            arrays = _read_scale_checkpoint(
                checkpoint,
                replication_id=replication_id,
                experiment_seed=protocol.scale_calibration_seed,
                expected_points=expected_points,
            )
        else:
            if checkpoint.exists() and not resume:
                checkpoint.unlink()
            components = no_social_component_path(
                experiment_seed=protocol.scale_calibration_seed,
                replication_id=replication_id,
                baseline=baseline,
                protocol=protocol,
                adaptive_attention=False,
            )
            arrays = _component_arrays(components)
            _write_scale_checkpoint(
                checkpoint,
                replication_id=replication_id,
                experiment_seed=protocol.scale_calibration_seed,
                return_values=arrays[0],
                belief_values=arrays[1],
                flow_values=arrays[2],
            )
        all_return.append(arrays[0])
        all_belief.append(arrays[1])
        all_flow.append(arrays[2])
        if progress is not None:
            progress("scale", replication_id + 1, protocol.n_scale_replications, reused)

    c_ret = float(np.median(np.concatenate(all_return)))
    c_bel = float(np.median(np.concatenate(all_belief)))
    c_flow = float(np.median(np.concatenate(all_flow)))
    scales = CIDReferenceScales(
        return_scale=c_ret,
        belief_scale=c_bel,
        order_flow_scale=c_flow,
    )
    _atomic_json(
        output_dir / _SCALE_ARTIFACT_NAME,
        {
            "final_calibration": True,
            "stage": "scale",
            "calibration_alpha": 0.0,
            "scale_seed": protocol.scale_calibration_seed,
            "n_scale_replications": protocol.n_scale_replications,
            "rolling_points_per_run": expected_points,
            "reference_scales": {"c_ret": c_ret, "c_bel": c_bel, "c_F": c_flow},
        },
    )
    return scales


def load_reference_scales(output_dir: str | Path) -> CIDReferenceScales:
    """Load and validate the completed scale-stage artifact."""

    path = Path(output_dir) / _SCALE_ARTIFACT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["reference_scales"]
    return CIDReferenceScales(
        return_scale=values["c_ret"],
        belief_scale=values["c_bel"],
        order_flow_scale=values["c_F"],
    )


def run_threshold_calibration_stage(
    *,
    output_dir: str | Path,
    scales: CIDReferenceScales | None = None,
    protocol: MarketEvaluationCalibrationProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> FinalMarketCalibrationRun:
    """Run/resume the independent threshold sample and write the final artifact."""

    protocol = protocol or first_market_evaluation_calibration_protocol()
    baseline = baseline or first_refined_baseline_specification()
    _validate_inputs(protocol, baseline)
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")
    output_dir = Path(output_dir)
    if scales is None:
        scales = load_reference_scales(output_dir)
    if not isinstance(scales, CIDReferenceScales):
        raise TypeError("scales must be CIDReferenceScales")

    peaks: list[float] = []
    for replication_id in range(protocol.n_threshold_replications):
        checkpoint = _threshold_checkpoint_path(output_dir, replication_id)
        reused = checkpoint.exists() and resume
        if reused:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload["replication_id"] != replication_id:
                raise ValueError(f"threshold checkpoint replication mismatch: {checkpoint}")
            if payload["experiment_seed"] != protocol.threshold_calibration_seed:
                raise ValueError(f"threshold checkpoint seed mismatch: {checkpoint}")
            peak = float(payload["peak_cid"])
        else:
            if checkpoint.exists() and not resume:
                checkpoint.unlink()
            components = no_social_component_path(
                experiment_seed=protocol.threshold_calibration_seed,
                replication_id=replication_id,
                baseline=baseline,
                protocol=protocol,
                adaptive_attention=False,
            )
            cid_path = standardise_cid_components(
                components,
                scales=scales,
                weights=protocol.cid_weights,
            )
            peak = float(max(point.cid for point in cid_path))
            _atomic_json(
                checkpoint,
                {
                    "replication_id": replication_id,
                    "experiment_seed": protocol.threshold_calibration_seed,
                    "peak_cid": peak,
                },
            )
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError(f"invalid threshold peak CID in replication {replication_id}")
        peaks.append(peak)
        if progress is not None:
            progress("threshold", replication_id + 1, protocol.n_threshold_replications, reused)

    threshold = float(
        np.quantile(
            np.asarray(peaks, dtype=float),
            protocol.cid_peak_quantile,
            method=_QUANTILE_METHOD,
        )
    )
    calibration = MarketEvaluationCalibration(
        protocol=protocol,
        reference_scales=scales,
        cid_weights=protocol.cid_weights,
        cid_threshold=threshold,
    )

    csv_path = output_dir / _THRESHOLD_CSV_NAME
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["replication_id", "peak_cid"])
        for replication_id, peak in enumerate(peaks):
            writer.writerow([replication_id, f"{peak:.17g}"])
    os.replace(temporary_csv, csv_path)

    weights = protocol.cid_weights
    final_payload = {
        "purpose": "final frozen-method D042 no-social market-evaluation calibration",
        "final_calibration": True,
        "calibration_alpha": 0.0,
        "adaptive_attention_during_calibration": False,
        "adaptive_attention_note": (
            "At alpha=0, fixed and adaptive attention are exactly equivalent for market paths "
            "and CID components; production uses the cheaper fixed-attention path."
        ),
        "protocol": {
            "scale_seed": protocol.scale_calibration_seed,
            "threshold_seed": protocol.threshold_calibration_seed,
            "n_scale_replications": protocol.n_scale_replications,
            "n_threshold_replications": protocol.n_threshold_replications,
            "horizon": protocol.horizon,
            "burn_in": protocol.burn_in,
            "rolling_window": protocol.rolling_window,
            "rolling_points_per_run": protocol.expected_rolling_points_per_run,
            "cid_peak_quantile": protocol.cid_peak_quantile,
            "quantile_method": _QUANTILE_METHOD,
            "stabilisation_length": protocol.stabilisation_length,
        },
        "frozen_baseline": {
            "n_agents": baseline.n_agents,
            "k": baseline.k,
            "horizon": baseline.horizon,
            "hub_q": baseline.hub_q,
            "p_sw": baseline.p_sw,
            "a0": baseline.a0,
            "parameters": asdict(baseline.parameters),
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
        "c_CID": threshold,
        "component_guardrails": {
            "return": None,
            "belief": None,
            "order_flow": None,
        },
    }
    _atomic_json(output_dir / _FINAL_ARTIFACT_NAME, final_payload)

    return FinalMarketCalibrationRun(
        protocol=protocol,
        baseline=baseline,
        calibration=calibration,
        threshold_peak_cids=tuple(peaks),
        output_dir=output_dir,
    )


def run_final_market_calibration(
    *,
    output_dir: str | Path,
    protocol: MarketEvaluationCalibrationProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> FinalMarketCalibrationRun:
    """Run/resume both frozen D042 stages in the required order."""

    protocol = protocol or first_market_evaluation_calibration_protocol()
    baseline = baseline or first_refined_baseline_specification()
    scales = run_scale_calibration_stage(
        output_dir=output_dir,
        protocol=protocol,
        baseline=baseline,
        resume=resume,
        progress=progress,
    )
    return run_threshold_calibration_stage(
        output_dir=output_dir,
        scales=scales,
        protocol=protocol,
        baseline=baseline,
        resume=resume,
        progress=progress,
    )
