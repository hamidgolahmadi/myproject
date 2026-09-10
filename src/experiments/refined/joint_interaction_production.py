"""Resumable production/finalization layer for frozen D049 joint interactions."""

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
from .confirmatory_runner import ConfirmatoryTreatmentRecord
from .frozen_market_calibration import (
    FROZEN_CONFIGURATION_FINGERPRINT,
    FROZEN_REFERENCE_SCALES_FINGERPRINT,
    first_frozen_market_evaluation_calibration,
)
from .joint_interaction_analysis import (
    JointTreatmentRecord,
    analyse_joint_interaction_records,
)
from .joint_interaction_protocol import (
    JointInteractionProtocol,
    first_joint_interaction_protocol,
)
from .joint_interaction_runner import run_paired_joint_replication
from .market_calibration import MarketEvaluationCalibration


_JOINT_SCHEMA_VERSION = 1
_FINAL_RECORDS_NAME = "joint_records.csv"
_FINAL_METADATA_NAME = "joint_metadata.json"
_FINAL_ANALYSIS_NAME = "joint_analysis.json"
_FINAL_MEANS_NAME = "joint_topology_means.csv"
_FINAL_GAPS_NAME = "joint_topology_gaps.csv"
_FINAL_CONTRASTS_NAME = "joint_pairwise_contrasts.csv"
_FINAL_INTERACTIONS_NAME = "joint_interactions.csv"
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


def _flat_record(record: JointTreatmentRecord) -> dict:
    return {
        "cell_id": record.cell_id,
        "cell_kind": record.cell_kind,
        "alpha_cell": record.alpha,
        "beta": record.beta,
        "gamma_R": record.gamma_R,
        "mean_raw_local_reputation_std": record.mean_raw_local_reputation_std,
        "mean_raw_local_reputation_std_over_sigma0": (
            record.mean_raw_local_reputation_std_over_sigma0
        ),
        **asdict(record.treatment),
    }


def _record_from_flat(payload: dict) -> JointTreatmentRecord:
    item = dict(payload)
    cell_id = str(item.pop("cell_id"))
    cell_kind = str(item.pop("cell_kind"))
    alpha = float(item.pop("alpha_cell"))
    beta = float(item.pop("beta"))
    gamma_R = float(item.pop("gamma_R"))
    raw_std = float(item.pop("mean_raw_local_reputation_std"))
    raw_ratio = float(item.pop("mean_raw_local_reputation_std_over_sigma0"))
    return JointTreatmentRecord(
        cell_id=cell_id,
        cell_kind=cell_kind,
        alpha=alpha,
        beta=beta,
        gamma_R=gamma_R,
        mean_raw_local_reputation_std=raw_std,
        mean_raw_local_reputation_std_over_sigma0=raw_ratio,
        treatment=ConfirmatoryTreatmentRecord(**item),
    )


def _validate_design(
    protocol: JointInteractionProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> None:
    if not isinstance(protocol, JointInteractionProtocol):
        raise TypeError("protocol must be JointInteractionProtocol")
    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if not isinstance(calibration, MarketEvaluationCalibration):
        raise TypeError("calibration must be MarketEvaluationCalibration")
    if baseline.horizon != calibration.protocol.horizon:
        raise ValueError("baseline and calibration horizons must match")
    labels = tuple(spec.topology_label for spec in baseline.topology_specifications)
    if labels != protocol.topology_labels:
        raise ValueError("baseline topology order must match D049")
    if baseline.parameters.alpha != 0.75:
        raise ValueError("D049 source baseline must retain frozen D043 alpha=0.75")
    if baseline.parameters.beta != 1.0:
        raise ValueError("D049 source baseline must retain frozen D043 beta=1.0")
    if baseline.parameters.gamma_R != 0.9:
        raise ValueError("D049 source baseline must retain frozen D043 gamma_R=0.9")
    if baseline.parameters.sigma_0 != 0.0005:
        raise ValueError("D049 source baseline must retain frozen D043 sigma_0=0.0005")


def _configuration_payload(
    protocol: JointInteractionProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> dict:
    return {
        "schema_version": _JOINT_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "joint_record_fields": (
            "cell_id",
            "cell_kind",
            "alpha_cell",
            "beta",
            "gamma_R",
            "mean_raw_local_reputation_std",
            "mean_raw_local_reputation_std_over_sigma0",
        )
        + _confirmatory_record_field_names(),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
    }


def joint_interaction_configuration_fingerprint(
    protocol: JointInteractionProtocol,
    baseline: RefinedBaselineSpecification,
    calibration: MarketEvaluationCalibration,
) -> str:
    _validate_design(protocol, baseline, calibration)
    encoded = json.dumps(
        _configuration_payload(protocol, baseline, calibration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_path(output_dir: Path, cell_index: int, replication_id: int) -> Path:
    return (
        output_dir
        / "checkpoints"
        / f"cell_{cell_index:02d}"
        / f"replication_{replication_id:04d}.json"
    )


def _checkpoint_payload(
    *,
    cell_index: int,
    replication_id: int,
    configuration_fingerprint: str,
    records: tuple[JointTreatmentRecord, ...],
) -> dict:
    first = records[0]
    return {
        "status": "complete",
        "schema_version": _JOINT_SCHEMA_VERSION,
        "cell_index": cell_index,
        "cell_id": first.cell_id,
        "cell_kind": first.cell_kind,
        "alpha": first.alpha,
        "beta": first.beta,
        "gamma_R": first.gamma_R,
        "replication_id": replication_id,
        "configuration_fingerprint": configuration_fingerprint,
        "records": [_flat_record(record) for record in records],
    }


def _read_complete_checkpoint(
    path: Path,
    *,
    cell_index: int,
    replication_id: int,
    configuration_fingerprint: str,
    protocol: JointInteractionProtocol,
) -> tuple[JointTreatmentRecord, ...]:
    cell = protocol.cells[cell_index]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _JOINT_SCHEMA_VERSION:
        raise RuntimeError(f"stale D049 checkpoint schema: {path}")
    if payload.get("cell_index") != cell_index or payload.get("cell_id") != cell.cell_id:
        raise RuntimeError(f"D049 checkpoint cell mismatch: {path}")
    if payload.get("cell_kind") != cell.cell_kind:
        raise RuntimeError(f"D049 checkpoint cell kind mismatch: {path}")
    for key, expected in (("alpha", cell.alpha), ("beta", cell.beta), ("gamma_R", cell.gamma_R)):
        if float(payload.get(key)) != expected:
            raise RuntimeError(f"D049 checkpoint {key} mismatch: {path}")
    if payload.get("replication_id") != replication_id:
        raise RuntimeError(f"D049 checkpoint replication mismatch: {path}")
    if payload.get("configuration_fingerprint") != configuration_fingerprint:
        raise RuntimeError(f"D049 checkpoint configuration mismatch: {path}")
    if payload.get("status") != "complete":
        raise RuntimeError(f"D049 checkpoint is not complete: {path}")

    records = tuple(_record_from_flat(item) for item in payload.get("records", []))
    if len(records) != len(protocol.topology_labels):
        raise RuntimeError(f"D049 checkpoint does not contain one topology triplet: {path}")
    if tuple(record.treatment.topology_label for record in records) != protocol.topology_labels:
        raise RuntimeError(f"D049 checkpoint topology order mismatch: {path}")
    if any(record.treatment.replication_id != replication_id for record in records):
        raise RuntimeError(f"D049 checkpoint contains wrong replication id: {path}")
    if any(record.treatment.experiment_seed != protocol.experiment_seed for record in records):
        raise RuntimeError(f"D049 checkpoint contains wrong experiment seed: {path}")
    if any(record.treatment.regime != "joint_interaction" for record in records):
        raise RuntimeError(f"D049 checkpoint contains a non-joint regime: {path}")
    for record in records:
        if (
            record.cell_id != cell.cell_id
            or record.cell_kind != cell.cell_kind
            or record.alpha != cell.alpha
            or record.beta != cell.beta
            or record.gamma_R != cell.gamma_R
        ):
            raise RuntimeError(f"D049 checkpoint record cell metadata mismatch: {path}")
    return records


def run_joint_interaction_range(
    *,
    cell_index: int,
    start_replication: int,
    stop_replication: int,
    output_dir: str | Path,
    resume: bool = True,
    protocol: JointInteractionProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[JointTreatmentRecord, ...]:
    """Run one D049 cell over a half-open replication range with checkpoints."""

    protocol = protocol or first_joint_interaction_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)

    if isinstance(cell_index, bool) or not isinstance(cell_index, int):
        raise TypeError("cell_index must be an integer")
    if not 0 <= cell_index < protocol.n_cells:
        raise ValueError("cell_index is outside the frozen D049 design")
    for name, value in (
        ("start_replication", start_replication),
        ("stop_replication", stop_replication),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 0 <= start_replication < stop_replication <= protocol.n_replications:
        raise ValueError("replication range must satisfy 0 <= start < stop <= n_replications")

    cell = protocol.cells[cell_index]
    output_dir = Path(output_dir)
    fingerprint = joint_interaction_configuration_fingerprint(protocol, baseline, calibration)
    collected: list[JointTreatmentRecord] = []
    total = stop_replication - start_replication

    for local_index, replication_id in enumerate(
        range(start_replication, stop_replication), start=1
    ):
        checkpoint = _checkpoint_path(output_dir, cell_index, replication_id)
        from_checkpoint = False
        if checkpoint.exists() and resume:
            records = _read_complete_checkpoint(
                checkpoint,
                cell_index=cell_index,
                replication_id=replication_id,
                configuration_fingerprint=fingerprint,
                protocol=protocol,
            )
            from_checkpoint = True
        else:
            records = run_paired_joint_replication(
                experiment_seed=protocol.experiment_seed,
                replication_id=replication_id,
                cell=cell,
                baseline=baseline,
                calibration=calibration,
            )
            _atomic_json(
                checkpoint,
                _checkpoint_payload(
                    cell_index=cell_index,
                    replication_id=replication_id,
                    configuration_fingerprint=fingerprint,
                    records=records,
                ),
            )
        collected.extend(records)
        if progress_callback is not None:
            progress_callback(local_index, total, from_checkpoint)

    return tuple(collected)


def load_all_joint_interaction_records(
    *,
    output_dir: str | Path,
    protocol: JointInteractionProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> tuple[JointTreatmentRecord, ...]:
    """Load the complete D049 design, refusing missing or stale checkpoints."""

    protocol = protocol or first_joint_interaction_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)
    fingerprint = joint_interaction_configuration_fingerprint(protocol, baseline, calibration)

    records: list[JointTreatmentRecord] = []
    missing: list[tuple[int, int]] = []
    for cell_index in range(protocol.n_cells):
        for replication_id in range(protocol.n_replications):
            checkpoint = _checkpoint_path(output_dir, cell_index, replication_id)
            if not checkpoint.exists():
                missing.append((cell_index, replication_id))
                continue
            records.extend(
                _read_complete_checkpoint(
                    checkpoint,
                    cell_index=cell_index,
                    replication_id=replication_id,
                    configuration_fingerprint=fingerprint,
                    protocol=protocol,
                )
            )
    if missing:
        preview = ", ".join(f"c{c}:r{r}" for c, r in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"cannot finalise D049: {len(missing)} cell/replication checkpoints are missing ({preview}{suffix})"
        )
    return tuple(records)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty D049 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finalize_joint_interaction_production(
    *,
    output_dir: str | Path,
    protocol: JointInteractionProtocol | None = None,
    baseline: RefinedBaselineSpecification | None = None,
    calibration: MarketEvaluationCalibration | None = None,
) -> dict[str, Path]:
    """Create final D049 cell-surface and interaction artifacts."""

    protocol = protocol or first_joint_interaction_protocol()
    baseline = baseline or first_refined_baseline_specification()
    calibration = calibration or first_frozen_market_evaluation_calibration()
    _validate_design(protocol, baseline, calibration)
    output_dir = Path(output_dir)

    records = load_all_joint_interaction_records(
        output_dir=output_dir,
        protocol=protocol,
        baseline=baseline,
        calibration=calibration,
    )
    analysis = analyse_joint_interaction_records(
        records,
        protocol=protocol,
        require_full_sample=True,
    )
    if not analysis.cross_cell_common_random_numbers_verified:
        raise RuntimeError("D049 cross-cell common-random-number validation failed")
    if not analysis.alpha_zero_economic_path_null_verified:
        raise RuntimeError("D049 alpha=0 economic-path topology null failed")

    fingerprint = joint_interaction_configuration_fingerprint(protocol, baseline, calibration)
    records_path = output_dir / _FINAL_RECORDS_NAME
    metadata_path = output_dir / _FINAL_METADATA_NAME
    analysis_path = output_dir / _FINAL_ANALYSIS_NAME
    means_path = output_dir / _FINAL_MEANS_NAME
    gaps_path = output_dir / _FINAL_GAPS_NAME
    contrasts_path = output_dir / _FINAL_CONTRASTS_NAME
    interactions_path = output_dir / _FINAL_INTERACTIONS_NAME

    _write_csv(records_path, [_flat_record(record) for record in records])
    _write_csv(means_path, [asdict(item) for item in analysis.topology_means])
    _write_csv(gaps_path, [asdict(item) for item in analysis.topology_gaps])
    _write_csv(contrasts_path, [asdict(item) for item in analysis.pairwise_contrasts])
    _write_csv(interactions_path, [asdict(item) for item in analysis.interactions])
    _atomic_json(analysis_path, asdict(analysis))

    metadata = {
        "purpose": "D049 sequential exploratory reduced-factorial alpha-beta-gamma_R interaction experiment",
        "final_joint_interaction": True,
        "confirmatory": False,
        "configuration_fingerprint": fingerprint,
        "protocol": asdict(protocol),
        "baseline": asdict(baseline),
        "calibration": asdict(calibration),
        "frozen_d042_d043_configuration_fingerprint": FROZEN_CONFIGURATION_FINGERPRINT,
        "frozen_reference_scales_fingerprint": FROZEN_REFERENCE_SCALES_FINGERPRINT,
        "n_complete_replications_per_cell": protocol.n_replications,
        "n_cells": protocol.n_cells,
        "n_factorial_cells": 48,
        "n_checkpoints": protocol.n_checkpoints,
        "n_treatment_records": len(records),
        "n_simulations": protocol.n_simulations,
        "bootstrap_design": "resample complete replication blocks containing all 50 cells and R/SW/SF treatments",
        "multiplicity": "none; D049 is sequential exploratory interaction mapping",
        "interaction_topology_contrast": "SF-SW",
        "alpha_zero_economic_path_null_verified": analysis.alpha_zero_economic_path_null_verified,
        "cross_cell_common_random_numbers_verified": analysis.cross_cell_common_random_numbers_verified,
        "sigma0_policy": "fixed at D043 value 0.0005; no gamma-dependent rescaling",
        "partial_results_guard": "final artifacts written only after every cell/replication checkpoint validates",
    }
    _atomic_json(metadata_path, metadata)

    return {
        "records": records_path,
        "metadata": metadata_path,
        "analysis": analysis_path,
        "means": means_path,
        "gaps": gaps_path,
        "contrasts": contrasts_path,
        "interactions": interactions_path,
    }
