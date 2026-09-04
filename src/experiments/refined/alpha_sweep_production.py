"""Resumable production layer for the frozen D046 exploratory alpha sweep.

Each checkpoint is one complete R/SW/SF triplet for one alpha value and one
replication id. Final curve artifacts are created only after the complete
alpha-by-replication design is present and validated.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from .alpha_sweep_analysis import analyse_alpha_sweep_records
from .alpha_sweep_protocol import AlphaSweepProtocol, first_alpha_sweep_protocol
from .baseline_specification import (
    RefinedBaselineSpecification,
    first_refined_baseline_specification,
)
from .confirmatory_runner import (
    ConfirmatoryTreatmentRecord,
    run_paired_confirmatory_replication,
)
from .frozen_market_calibration import (
    FROZEN_CONFIGURATION_FINGERPRINT,
    FROZEN_REFERENCE_SCALES_FINGERPRINT,
    first_frozen_market_evaluation_calibration,
)
from .market_calibration import MarketEvaluationCalibration


_ALPHA_SWEEP_SCHEMA_VERSION = 1
_FINAL_RECORDS_NAME = "alpha_sweep_records.csv"
_FINAL_METADATA_NAME = "alpha_sweep_metadata.json"
_FINAL_ANALYSIS_NAME = "alpha_sweep_analysis.json"
_FINAL_MEANS_NAME = "alpha_topology_means.csv"
_FINAL_GAPS_NAME = "alpha_topology_gaps.csv"
_FINAL_CONTRASTS_NAME = "alpha_pairwise_contrasts.csv"
ProgressCallback = Callable[[int, int, bool], None]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _record_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(ConfirmatoryTreatmentRecord))


def _validate_design(
    protocol: AlphaSweepProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> None:
    if not isinstance(protocol, AlphaSweepProtocol):
        raise TypeError("protocol must be AlphaSweepProtocol")
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if not isinstance(calibration, MarketEvaluationCalibration):
        raise TypeError("calibration must be MarketEvaluationCalibration")
    if baseline.horizon != calibration.protocol.horizon:
        raise ValueError("baseline and calibration horizons must match")
    labels = tuple(spec.topology_label for spec in baseline.topology_specifications)
    if labels != protocol.topology_labels:
        raise ValueError("baseline topology order must match D046")
    if baseline.parameters.alpha != 0.75:
        raise ValueError("D046 requires the frozen D043 alpha=0.75 baseline anchor")


def _configuration_payload(
    protocol: AlphaSweepProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> dict:
    return {
        "schema_version": _ALPHA_SWEEP_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "record_fields": _record_field_names(),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
    }


def alpha_sweep_configuration_fingerprint(
    protocol: AlphaSweepProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> str:
    """Return a stable fingerprint of every D046 production-defining input."""

    _validate_design(protocol, baseline, calibration)
    encoded = json.dumps(
        _configuration_payload(protocol, baseline, calibration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(output_dir: Path, alpha_index: int, replication_id: int) -> Path:
    return (
        output_dir
        / "checkpoints"
        / f"alpha_{alpha_index:02d}"
        / f"replication_{replication_id:04d}.json"
    )


def _checkpoint_payload(
    *,
    alpha_index: int,
    alpha: float,
    replication_id: int,
    configuration_fingerprint: str,
    records: tuple[ConfirmatoryTreatmentRecord, ...],
) -> dict:
    return {
        "status": "complete",
        "schema_version": _ALPHA_SWEEP_SCHEMA_VERSION,
        "alpha_index": alpha_index,
        "alpha": alpha,
        "replication_id": replication_id,
        "configuration_fingerprint": configuration_fingerprint,
        "records": [asdict(record) for record in records],
    }


def _read_complete_checkpoint(
    path: Path,
    *,
    alpha_index: int,
    alpha: float,
    replication_id: int,
    configuration_fingerprint: str,
    protocol: AlphaSweepProtocol,
) -> tuple[ConfirmatoryTreatmentRecord, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _ALPHA_SWEEP_SCHEMA_VERSION:
        raise RuntimeError(f"stale D046 checkpoint schema: {path}")
    if payload.get("alpha_index") != alpha_index or float(payload.get("alpha")) != alpha:
        raise RuntimeError(f"D046 checkpoint alpha mismatch: {path}")
    if payload.get("replication_id") != replication_id:
        raise RuntimeError(f"D046 checkpoint replication mismatch: {path}")
    if payload.get("configuration_fingerprint") != configuration_fingerprint:
        raise RuntimeError(f"D046 checkpoint configuration mismatch: {path}")
    if payload.get("status") != "complete":
        raise RuntimeError(f"D046 checkpoint is not complete: {path}")

    records = tuple(ConfirmatoryTreatmentRecord(**item) for item in payload.get("records", []))
    if len(records) != len(protocol.topology_labels):
        raise RuntimeError(f"D046 checkpoint does not contain one topology triplet: {path}")
    if tuple(record.topology_label for record in records) != protocol.topology_labels:
        raise RuntimeError(f"D046 checkpoint topology order mismatch: {path}")
    if any(record.replication_id != replication_id for record in records):
        raise RuntimeError(f"D046 checkpoint contains wrong replication id: {path}")
    if any(record.experiment_seed != protocol.experiment_seed for record in records):
        raise RuntimeError(f"D046 checkpoint contains wrong experiment seed: {path}")
    if any(record.regime != "alpha_sweep" for record in records):
        raise RuntimeError(f"D046 checkpoint contains a non-alpha-sweep regime: {path}")
    if any(float(record.alpha) != alpha for record in records):
        raise RuntimeError(f"D046 checkpoint treatment alpha mismatch: {path}")
    return records


def run_alpha_sweep_range(
    *,
    alpha_index: int,
    start_replication: int,
    stop_replication: int,
    output_dir: str | Path,
    resume: bool = True,
    protocol: AlphaSweepProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[ConfirmatoryTreatmentRecord, ...]:
    """Run one alpha and a half-open replication range with resumable checkpoints."""

    protocol = protocol or first_alpha_sweep_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)

    if isinstance(alpha_index, bool) or not isinstance(alpha_index, int):
        raise TypeError("alpha_index must be an integer")
    if not 0 <= alpha_index < protocol.n_alpha:
        raise ValueError("alpha_index is outside the frozen D046 grid")
    for name, value in (
        ("start_replication", start_replication),
        ("stop_replication", stop_replication),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 0 <= start_replication < stop_replication <= protocol.n_replications:
        raise ValueError("replication range must satisfy 0 <= start < stop <= n_replications")

    alpha = protocol.alpha_grid[alpha_index]
    output_dir = Path(output_dir)
    fingerprint = alpha_sweep_configuration_fingerprint(protocol, baseline, calibration)
    collected: list[ConfirmatoryTreatmentRecord] = []
    total = stop_replication - start_replication

    for local_index, replication_id in enumerate(
        range(start_replication, stop_replication), start=1
    ):
        checkpoint = _checkpoint_path(output_dir, alpha_index, replication_id)
        from_checkpoint = False
        if checkpoint.exists() and resume:
            records = _read_complete_checkpoint(
                checkpoint,
                alpha_index=alpha_index,
                alpha=alpha,
                replication_id=replication_id,
                configuration_fingerprint=fingerprint,
                protocol=protocol,
            )
            from_checkpoint = True
        else:
            records = run_paired_confirmatory_replication(
                experiment_seed=protocol.experiment_seed,
                replication_id=replication_id,
                regime="alpha_sweep",
                alpha_override=alpha,
                baseline=baseline,
                calibration=calibration,
            )
            _atomic_json(
                checkpoint,
                _checkpoint_payload(
                    alpha_index=alpha_index,
                    alpha=alpha,
                    replication_id=replication_id,
                    configuration_fingerprint=fingerprint,
                    records=records,
                ),
            )

        collected.extend(records)
        if progress_callback is not None:
            progress_callback(local_index, total, from_checkpoint)

    return tuple(collected)


def load_all_alpha_sweep_records(
    *,
    output_dir: str | Path,
    protocol: AlphaSweepProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> tuple[ConfirmatoryTreatmentRecord, ...]:
    """Load the complete D046 design, refusing any missing or stale block."""

    protocol = protocol or first_alpha_sweep_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)
    fingerprint = alpha_sweep_configuration_fingerprint(protocol, baseline, calibration)

    records: list[ConfirmatoryTreatmentRecord] = []
    missing: list[tuple[int, int]] = []
    for alpha_index, alpha in enumerate(protocol.alpha_grid):
        for replication_id in range(protocol.n_replications):
            checkpoint = _checkpoint_path(output_dir, alpha_index, replication_id)
            if not checkpoint.exists():
                missing.append((alpha_index, replication_id))
                continue
            records.extend(
                _read_complete_checkpoint(
                    checkpoint,
                    alpha_index=alpha_index,
                    alpha=alpha,
                    replication_id=replication_id,
                    configuration_fingerprint=fingerprint,
                    protocol=protocol,
                )
            )

    if missing:
        preview = ", ".join(f"a{a}:r{r}" for a, r in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"cannot finalise D046: {len(missing)} alpha/replication checkpoints are missing ({preview}{suffix})"
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


def finalize_alpha_sweep_production(
    *,
    output_dir: str | Path,
    protocol: AlphaSweepProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> dict[str, Path]:
    """Create final D046 exploratory curve artifacts from the complete design."""

    protocol = protocol or first_alpha_sweep_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)

    records = load_all_alpha_sweep_records(
        output_dir=output_dir,
        protocol=protocol,
        baseline=baseline,
        calibration=calibration,
    )
    analysis = analyse_alpha_sweep_records(
        records,
        protocol=protocol,
        require_full_sample=True,
    )
    fingerprint = alpha_sweep_configuration_fingerprint(protocol, baseline, calibration)

    records_path = output_dir / _FINAL_RECORDS_NAME
    metadata_path = output_dir / _FINAL_METADATA_NAME
    analysis_path = output_dir / _FINAL_ANALYSIS_NAME
    means_path = output_dir / _FINAL_MEANS_NAME
    gaps_path = output_dir / _FINAL_GAPS_NAME
    contrasts_path = output_dir / _FINAL_CONTRASTS_NAME

    _write_csv(records_path, [asdict(record) for record in records])
    _write_csv(means_path, [asdict(item) for item in analysis.topology_means])
    _write_csv(gaps_path, [asdict(item) for item in analysis.topology_gaps])
    _write_csv(contrasts_path, [asdict(item) for item in analysis.pairwise_contrasts])
    _atomic_json(analysis_path, asdict(analysis))

    metadata = {
        "purpose": "D046 exploratory OAT alpha sweep after verified D045",
        "final_alpha_sweep": True,
        "confirmatory": False,
        "configuration_fingerprint": fingerprint,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
        "n_complete_replications_per_alpha": protocol.n_replications,
        "n_alpha": protocol.n_alpha,
        "n_treatment_records": len(records),
        "bootstrap_design": "resample complete replication blocks containing all alpha values and R/SW/SF treatments",
        "multiplicity": "none; D046 is exploratory curve/regime mapping",
        "alpha_zero_negative_control": "exact economic-path topology null is required during final analysis",
        "partial_results_guard": "final artifacts are written only after every alpha/replication checkpoint is present",
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
