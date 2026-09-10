"""Resumable production layer for the frozen D048 exploratory gamma_R sweep.

Each checkpoint is one complete R/SW/SF triplet for one gamma_R value and one
replication id. Final curve artifacts are created only after the complete
gamma-by-replication design is present and validated.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from .baseline_specification import (
    RefinedBaselineSpecification,
    first_refined_baseline_specification,
)
from .confirmatory_runner import ConfirmatoryTreatmentRecord
from .frozen_market_calibration import (
    FROZEN_CONFIGURATION_FINGERPRINT,
    FROZEN_REFERENCE_SCALES_FINGERPRINT,
    first_frozen_market_evaluation_calibration,
)
from .gamma_sweep_analysis import (
    GammaSweepTreatmentRecord,
    analyse_gamma_sweep_records,
)
from .gamma_sweep_protocol import GammaSweepProtocol, first_gamma_sweep_protocol
from .gamma_sweep_runner import run_paired_gamma_replication
from .market_calibration import MarketEvaluationCalibration


_GAMMA_SWEEP_SCHEMA_VERSION = 1
_FINAL_RECORDS_NAME = "gamma_sweep_records.csv"
_FINAL_METADATA_NAME = "gamma_sweep_metadata.json"
_FINAL_ANALYSIS_NAME = "gamma_sweep_analysis.json"
_FINAL_MEANS_NAME = "gamma_topology_means.csv"
_FINAL_GAPS_NAME = "gamma_topology_gaps.csv"
_FINAL_CONTRASTS_NAME = "gamma_pairwise_contrasts.csv"
ProgressCallback = Callable[[int, int, bool], None]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _confirmatory_record_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(ConfirmatoryTreatmentRecord))


def _flat_record(record: GammaSweepTreatmentRecord) -> dict:
    return {
        "gamma_R": record.gamma_R,
        "mean_raw_local_reputation_std": record.mean_raw_local_reputation_std,
        "mean_raw_local_reputation_std_over_sigma0": (
            record.mean_raw_local_reputation_std_over_sigma0
        ),
        **asdict(record.treatment),
    }


def _record_from_flat(payload: dict) -> GammaSweepTreatmentRecord:
    item = dict(payload)
    gamma_R = float(item.pop("gamma_R"))
    raw_std = float(item.pop("mean_raw_local_reputation_std"))
    raw_ratio = float(item.pop("mean_raw_local_reputation_std_over_sigma0"))
    return GammaSweepTreatmentRecord(
        gamma_R=gamma_R,
        mean_raw_local_reputation_std=raw_std,
        mean_raw_local_reputation_std_over_sigma0=raw_ratio,
        treatment=ConfirmatoryTreatmentRecord(**item),
    )


def _validate_design(
    protocol: GammaSweepProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> None:
    if not isinstance(protocol, GammaSweepProtocol):
        raise TypeError("protocol must be GammaSweepProtocol")
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if not isinstance(calibration, MarketEvaluationCalibration):
        raise TypeError("calibration must be MarketEvaluationCalibration")
    if baseline.horizon != calibration.protocol.horizon:
        raise ValueError("baseline and calibration horizons must match")
    labels = tuple(spec.topology_label for spec in baseline.topology_specifications)
    if labels != protocol.topology_labels:
        raise ValueError("baseline topology order must match D048")
    if baseline.parameters.alpha != 0.75:
        raise ValueError("D048 source baseline must retain frozen D043 alpha=0.75")
    if baseline.parameters.beta != 1.0:
        raise ValueError("D048 source baseline must retain frozen D043 beta=1.0")
    if baseline.parameters.gamma_R != 0.9:
        raise ValueError("D048 source baseline must retain frozen D043 gamma_R=0.9")


def _configuration_payload(
    protocol: GammaSweepProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> dict:
    return {
        "schema_version": _GAMMA_SWEEP_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "gamma_record_fields": (
            "gamma_R",
            "mean_raw_local_reputation_std",
            "mean_raw_local_reputation_std_over_sigma0",
        )
        + _confirmatory_record_field_names(),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
    }


def gamma_sweep_configuration_fingerprint(
    protocol: GammaSweepProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> str:
    """Return a stable fingerprint of every D048 production-defining input."""

    _validate_design(protocol, baseline, calibration)
    encoded = json.dumps(
        _configuration_payload(protocol, baseline, calibration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(output_dir: Path, gamma_index: int, replication_id: int) -> Path:
    return (
        output_dir
        / "checkpoints"
        / f"gamma_{gamma_index:02d}"
        / f"replication_{replication_id:04d}.json"
    )


def _checkpoint_payload(
    *,
    gamma_index: int,
    gamma_R: float,
    replication_id: int,
    configuration_fingerprint: str,
    records: tuple[GammaSweepTreatmentRecord, ...],
) -> dict:
    return {
        "status": "complete",
        "schema_version": _GAMMA_SWEEP_SCHEMA_VERSION,
        "gamma_index": gamma_index,
        "gamma_R": gamma_R,
        "replication_id": replication_id,
        "configuration_fingerprint": configuration_fingerprint,
        "records": [_flat_record(record) for record in records],
    }


def _read_complete_checkpoint(
    path: Path,
    *,
    gamma_index: int,
    gamma_R: float,
    replication_id: int,
    configuration_fingerprint: str,
    protocol: GammaSweepProtocol,
) -> tuple[GammaSweepTreatmentRecord, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _GAMMA_SWEEP_SCHEMA_VERSION:
        raise RuntimeError(f"stale D048 checkpoint schema: {path}")
    if payload.get("gamma_index") != gamma_index or float(payload.get("gamma_R")) != gamma_R:
        raise RuntimeError(f"D048 checkpoint gamma_R mismatch: {path}")
    if payload.get("replication_id") != replication_id:
        raise RuntimeError(f"D048 checkpoint replication mismatch: {path}")
    if payload.get("configuration_fingerprint") != configuration_fingerprint:
        raise RuntimeError(f"D048 checkpoint configuration mismatch: {path}")
    if payload.get("status") != "complete":
        raise RuntimeError(f"D048 checkpoint is not complete: {path}")

    records = tuple(_record_from_flat(item) for item in payload.get("records", []))
    if len(records) != len(protocol.topology_labels):
        raise RuntimeError(f"D048 checkpoint does not contain one topology triplet: {path}")
    if tuple(record.treatment.topology_label for record in records) != protocol.topology_labels:
        raise RuntimeError(f"D048 checkpoint topology order mismatch: {path}")
    if any(record.treatment.replication_id != replication_id for record in records):
        raise RuntimeError(f"D048 checkpoint contains wrong replication id: {path}")
    if any(record.treatment.experiment_seed != protocol.experiment_seed for record in records):
        raise RuntimeError(f"D048 checkpoint contains wrong experiment seed: {path}")
    if any(record.treatment.regime != "gamma_sweep" for record in records):
        raise RuntimeError(f"D048 checkpoint contains a non-gamma-sweep regime: {path}")
    if any(record.gamma_R != gamma_R for record in records):
        raise RuntimeError(f"D048 checkpoint treatment gamma_R mismatch: {path}")
    if any(float(record.treatment.alpha) != protocol.alpha_anchor for record in records):
        raise RuntimeError(f"D048 checkpoint treatment alpha mismatch: {path}")
    return records


def _sweep_baseline(
    baseline: RefinedBaselineSpecification,
    *,
    alpha: float,
    beta: float,
    gamma_R: float,
) -> RefinedBaselineSpecification:
    parameters = replace(
        baseline.parameters,
        alpha=alpha,
        beta=beta,
        gamma_R=gamma_R,
    )
    return replace(baseline, parameters=parameters)


def run_gamma_sweep_range(
    *,
    gamma_index: int,
    start_replication: int,
    stop_replication: int,
    output_dir: str | Path,
    resume: bool = True,
    protocol: GammaSweepProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[GammaSweepTreatmentRecord, ...]:
    """Run one gamma_R and a half-open replication range with checkpoints."""

    protocol = protocol or first_gamma_sweep_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)

    if isinstance(gamma_index, bool) or not isinstance(gamma_index, int):
        raise TypeError("gamma_index must be an integer")
    if not 0 <= gamma_index < protocol.n_gamma:
        raise ValueError("gamma_index is outside the frozen D048 grid")
    for name, value in (
        ("start_replication", start_replication),
        ("stop_replication", stop_replication),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 0 <= start_replication < stop_replication <= protocol.n_replications:
        raise ValueError("replication range must satisfy 0 <= start < stop <= n_replications")

    gamma_R = protocol.gamma_grid[gamma_index]
    output_dir = Path(output_dir)
    fingerprint = gamma_sweep_configuration_fingerprint(protocol, baseline, calibration)
    treatment_baseline = _sweep_baseline(
        baseline,
        alpha=protocol.alpha_anchor,
        beta=protocol.beta_anchor,
        gamma_R=gamma_R,
    )
    collected: list[GammaSweepTreatmentRecord] = []
    total = stop_replication - start_replication

    for local_index, replication_id in enumerate(
        range(start_replication, stop_replication), start=1
    ):
        checkpoint = _checkpoint_path(output_dir, gamma_index, replication_id)
        from_checkpoint = False
        if checkpoint.exists() and resume:
            records = _read_complete_checkpoint(
                checkpoint,
                gamma_index=gamma_index,
                gamma_R=gamma_R,
                replication_id=replication_id,
                configuration_fingerprint=fingerprint,
                protocol=protocol,
            )
            from_checkpoint = True
        else:
            records = run_paired_gamma_replication(
                experiment_seed=protocol.experiment_seed,
                replication_id=replication_id,
                baseline=treatment_baseline,
                calibration=calibration,
            )
            _atomic_json(
                checkpoint,
                _checkpoint_payload(
                    gamma_index=gamma_index,
                    gamma_R=gamma_R,
                    replication_id=replication_id,
                    configuration_fingerprint=fingerprint,
                    records=records,
                ),
            )
        collected.extend(records)
        if progress_callback is not None:
            progress_callback(local_index, total, from_checkpoint)

    return tuple(collected)


def load_all_gamma_sweep_records(
    *,
    output_dir: str | Path,
    protocol: GammaSweepProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> tuple[GammaSweepTreatmentRecord, ...]:
    """Load the complete D048 design, refusing any missing or stale block."""

    protocol = protocol or first_gamma_sweep_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)
    fingerprint = gamma_sweep_configuration_fingerprint(protocol, baseline, calibration)

    records: list[GammaSweepTreatmentRecord] = []
    missing: list[tuple[int, int]] = []
    for gamma_index, gamma_R in enumerate(protocol.gamma_grid):
        for replication_id in range(protocol.n_replications):
            checkpoint = _checkpoint_path(output_dir, gamma_index, replication_id)
            if not checkpoint.exists():
                missing.append((gamma_index, replication_id))
                continue
            records.extend(
                _read_complete_checkpoint(
                    checkpoint,
                    gamma_index=gamma_index,
                    gamma_R=gamma_R,
                    replication_id=replication_id,
                    configuration_fingerprint=fingerprint,
                    protocol=protocol,
                )
            )

    if missing:
        preview = ", ".join(f"g{g}:r{r}" for g, r in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"cannot finalise D048: {len(missing)} gamma/replication checkpoints are missing ({preview}{suffix})"
        )
    return tuple(records)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finalize_gamma_sweep_production(
    *,
    output_dir: str | Path,
    protocol: GammaSweepProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> dict[str, Path]:
    """Create final D048 exploratory curve artifacts from the complete design."""

    protocol = protocol or first_gamma_sweep_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)

    records = load_all_gamma_sweep_records(
        output_dir=output_dir,
        protocol=protocol,
        baseline=baseline,
        calibration=calibration,
    )
    analysis = analyse_gamma_sweep_records(
        records,
        protocol=protocol,
        require_full_sample=True,
    )
    fingerprint = gamma_sweep_configuration_fingerprint(protocol, baseline, calibration)

    records_path = output_dir / _FINAL_RECORDS_NAME
    metadata_path = output_dir / _FINAL_METADATA_NAME
    analysis_path = output_dir / _FINAL_ANALYSIS_NAME
    means_path = output_dir / _FINAL_MEANS_NAME
    gaps_path = output_dir / _FINAL_GAPS_NAME
    contrasts_path = output_dir / _FINAL_CONTRASTS_NAME

    _write_csv(records_path, [_flat_record(record) for record in records])
    _write_csv(means_path, [asdict(item) for item in analysis.topology_means])
    _write_csv(gaps_path, [asdict(item) for item in analysis.topology_gaps])
    _write_csv(contrasts_path, [asdict(item) for item in analysis.pairwise_contrasts])
    _atomic_json(analysis_path, asdict(analysis))

    metadata = {
        "purpose": "D048 exploratory OAT gamma_R sweep after completed D046/D047",
        "final_gamma_sweep": True,
        "confirmatory": False,
        "configuration_fingerprint": fingerprint,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
        "d046_selected_alpha_anchor": protocol.alpha_anchor,
        "d047_selected_beta_anchor": protocol.beta_anchor,
        "n_complete_replications_per_gamma": protocol.n_replications,
        "n_gamma": protocol.n_gamma,
        "n_treatment_records": len(records),
        "bootstrap_design": "resample complete replication blocks containing all gamma_R values and R/SW/SF treatments",
        "multiplicity": "none; D048 is exploratory OAT curve/regime mapping",
        "gamma_zero_control": "gamma_R=0 removes reputation memory but not performance-based reputation",
        "gamma_high_boundary": "gamma_R=0.999 is a horizon-scale persistence stress point, not a no-learning control",
        "reputation_scale_diagnostics": [
            "mean_raw_local_reputation_std",
            "mean_raw_local_reputation_std_over_sigma0",
        ],
        "partial_results_guard": "final artifacts are written only after every gamma/replication checkpoint is present",
    }
    _atomic_json(metadata_path, metadata)

    return {
        "records": records_path,
        "metadata": metadata_path,
        "analysis": analysis_path,
        "means": means_path,
        "gaps": gaps_path,
        "contrasts": contrasts_path,
    }
