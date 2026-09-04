"""Resumable production layer for the frozen D045 confirmatory experiment.

Each paired replication is checkpointed as one indivisible R/SW/SF triplet.
The final CSV and bootstrap inference artifacts are created only when every
predeclared replication is present and valid.  Partial production runs therefore
cannot silently become a final confirmatory analysis.
"""

from __future__ import annotations

from dataclasses import asdict, fields
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
from .confirmatory_inference import analyse_confirmatory_records
from .confirmatory_protocol import (
    ConfirmatoryProductionProtocol,
    first_confirmatory_production_protocol,
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


_PRODUCTION_SCHEMA_VERSION = 1
_FINAL_RECORDS_NAME = "confirmatory_records.csv"
_FINAL_METADATA_NAME = "confirmatory_metadata.json"
_FINAL_ANALYSIS_NAME = "confirmatory_analysis.json"
_FINAL_MEANS_NAME = "topology_means.csv"
_FINAL_GAPS_NAME = "topology_gaps.csv"
_FINAL_CONTRASTS_NAME = "pairwise_contrasts.csv"
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


def _configuration_payload(
    protocol: ConfirmatoryProductionProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> dict:
    return {
        "schema_version": _PRODUCTION_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "record_fields": _record_field_names(),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
    }


def confirmatory_configuration_fingerprint(
    protocol: ConfirmatoryProductionProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> str:
    """Return a stable fingerprint of every D045 production-defining input."""

    _validate_design(protocol, baseline, calibration)
    encoded = json.dumps(
        _configuration_payload(protocol, baseline, calibration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_design(
    protocol: ConfirmatoryProductionProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> None:
    if not isinstance(protocol, ConfirmatoryProductionProtocol):
        raise TypeError("protocol must be ConfirmatoryProductionProtocol")
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if not isinstance(calibration, MarketEvaluationCalibration):
        raise TypeError("calibration must be MarketEvaluationCalibration")
    if baseline.horizon != calibration.protocol.horizon:
        raise ValueError("baseline and calibration horizons must match")
    labels = tuple(spec.topology_label for spec in baseline.topology_specifications)
    if labels != protocol.topology_labels:
        raise ValueError("baseline topology order must match the D045 protocol")


def _checkpoint_path(output_dir: Path, replication_id: int) -> Path:
    return output_dir / "replications" / f"replication_{replication_id:04d}.json"


def _checkpoint_payload(
    *,
    replication_id: int,
    configuration_fingerprint: str,
    records: tuple[ConfirmatoryTreatmentRecord, ...],
) -> dict:
    return {
        "status": "complete",
        "schema_version": _PRODUCTION_SCHEMA_VERSION,
        "replication_id": replication_id,
        "configuration_fingerprint": configuration_fingerprint,
        "records": [asdict(record) for record in records],
    }


def _read_complete_checkpoint(
    path: Path,
    *,
    replication_id: int,
    configuration_fingerprint: str,
    protocol: ConfirmatoryProductionProtocol,
) -> tuple[ConfirmatoryTreatmentRecord, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _PRODUCTION_SCHEMA_VERSION:
        raise RuntimeError(f"stale confirmatory checkpoint schema: {path}")
    if payload.get("replication_id") != replication_id:
        raise RuntimeError(f"confirmatory checkpoint replication mismatch: {path}")
    if payload.get("configuration_fingerprint") != configuration_fingerprint:
        raise RuntimeError(f"confirmatory checkpoint configuration mismatch: {path}")
    if payload.get("status") != "complete":
        raise RuntimeError(f"confirmatory checkpoint is not complete: {path}")

    records = tuple(ConfirmatoryTreatmentRecord(**item) for item in payload.get("records", []))
    if len(records) != len(protocol.topology_labels):
        raise RuntimeError(f"confirmatory checkpoint does not contain one topology triplet: {path}")
    if tuple(record.topology_label for record in records) != protocol.topology_labels:
        raise RuntimeError(f"confirmatory checkpoint topology order mismatch: {path}")
    if any(record.replication_id != replication_id for record in records):
        raise RuntimeError(f"confirmatory checkpoint contains wrong replication id: {path}")
    if any(record.experiment_seed != protocol.experiment_seed for record in records):
        raise RuntimeError(f"confirmatory checkpoint contains wrong experiment seed: {path}")
    if any(record.regime != "baseline" for record in records):
        raise RuntimeError(f"confirmatory production checkpoint must contain baseline records only: {path}")
    return records


def run_confirmatory_production_range(
    *,
    start_replication: int,
    stop_replication: int,
    output_dir: str | Path,
    resume: bool = True,
    protocol: ConfirmatoryProductionProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[ConfirmatoryTreatmentRecord, ...]:
    """Run a half-open range of paired baseline replications with checkpoints."""

    protocol = protocol or first_confirmatory_production_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)

    if isinstance(start_replication, bool) or not isinstance(start_replication, int):
        raise TypeError("start_replication must be an integer")
    if isinstance(stop_replication, bool) or not isinstance(stop_replication, int):
        raise TypeError("stop_replication must be an integer")
    if not 0 <= start_replication < stop_replication <= protocol.n_replications:
        raise ValueError("replication range must satisfy 0 <= start < stop <= n_replications")

    output_dir = Path(output_dir)
    fingerprint = confirmatory_configuration_fingerprint(protocol, baseline, calibration)
    collected: list[ConfirmatoryTreatmentRecord] = []
    total = stop_replication - start_replication

    for local_index, replication_id in enumerate(
        range(start_replication, stop_replication),
        start=1,
    ):
        checkpoint = _checkpoint_path(output_dir, replication_id)
        from_checkpoint = False
        if checkpoint.exists() and resume:
            records = _read_complete_checkpoint(
                checkpoint,
                replication_id=replication_id,
                configuration_fingerprint=fingerprint,
                protocol=protocol,
            )
            from_checkpoint = True
        else:
            records = run_paired_confirmatory_replication(
                experiment_seed=protocol.experiment_seed,
                replication_id=replication_id,
                regime="baseline",
                baseline=baseline,
                calibration=calibration,
            )
            _atomic_json(
                checkpoint,
                _checkpoint_payload(
                    replication_id=replication_id,
                    configuration_fingerprint=fingerprint,
                    records=records,
                ),
            )

        collected.extend(records)
        if progress_callback is not None:
            progress_callback(local_index, total, from_checkpoint)

    return tuple(collected)


def load_all_confirmatory_records(
    *,
    output_dir: str | Path,
    protocol: ConfirmatoryProductionProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> tuple[ConfirmatoryTreatmentRecord, ...]:
    """Load the full predeclared sample, refusing missing/stale checkpoints."""

    protocol = protocol or first_confirmatory_production_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)
    fingerprint = confirmatory_configuration_fingerprint(protocol, baseline, calibration)

    records: list[ConfirmatoryTreatmentRecord] = []
    missing: list[int] = []
    for replication_id in range(protocol.n_replications):
        checkpoint = _checkpoint_path(output_dir, replication_id)
        if not checkpoint.exists():
            missing.append(replication_id)
            continue
        records.extend(
            _read_complete_checkpoint(
                checkpoint,
                replication_id=replication_id,
                configuration_fingerprint=fingerprint,
                protocol=protocol,
            )
        )

    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"cannot finalise D045: {len(missing)} replication checkpoints are missing ({preview}{suffix})"
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


def finalize_confirmatory_production(
    *,
    output_dir: str | Path,
    protocol: ConfirmatoryProductionProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> dict[str, Path]:
    """Create final D045 records and inference artifacts only from all 1000 triplets."""

    protocol = protocol or first_confirmatory_production_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)

    records = load_all_confirmatory_records(
        output_dir=output_dir,
        protocol=protocol,
        baseline=baseline,
        calibration=calibration,
    )
    inference = analyse_confirmatory_records(
        records,
        protocol=protocol,
        require_full_sample=True,
    )
    fingerprint = confirmatory_configuration_fingerprint(protocol, baseline, calibration)

    records_path = output_dir / _FINAL_RECORDS_NAME
    means_path = output_dir / _FINAL_MEANS_NAME
    gaps_path = output_dir / _FINAL_GAPS_NAME
    contrasts_path = output_dir / _FINAL_CONTRASTS_NAME
    analysis_path = output_dir / _FINAL_ANALYSIS_NAME
    metadata_path = output_dir / _FINAL_METADATA_NAME

    _write_csv(records_path, [asdict(record) for record in records])
    _write_csv(means_path, [asdict(item) for item in inference.topology_means])
    _write_csv(gaps_path, [asdict(item) for item in inference.topology_gaps])
    _write_csv(contrasts_path, [asdict(item) for item in inference.pairwise_contrasts])
    _atomic_json(analysis_path, asdict(inference))

    metadata = {
        "purpose": "final D045 first confirmatory fixed-topology baseline experiment",
        "final_confirmatory": True,
        "configuration_fingerprint": fingerprint,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
        "n_complete_paired_replications": protocol.n_replications,
        "n_treatment_records": len(records),
        "bootstrap_design": "resample complete matched R/SW/SF replication triplets",
        "primary_multiplicity": "Holm FWER across all primary outcome x topology-pair hypotheses",
        "mechanism_multiplicity": "Holm FWER across all mechanism outcome x topology-pair hypotheses",
        "secondary_inference": "pointwise exploratory bootstrap intervals",
        "partial_results_guard": "final artifacts are written only after every predeclared checkpoint is present",
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
