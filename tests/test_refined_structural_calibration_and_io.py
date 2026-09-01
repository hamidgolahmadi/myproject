"""Tests for Decision D041 calibration and structural-validation persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.experiments.refined import (
    StructuralValidationCalibration,
    first_structural_validation_calibration,
    run_structural_ensemble,
    write_structural_result,
)


def calibration_kwargs() -> dict[str, object]:
    return {
        "experiment_seed": 20260901,
        "n_agents": 100,
        "k": 6,
        "n_replications": 1000,
        "hub_q": 5,
        "p_sw": 0.02,
        "a0": 1.0,
    }


def tiny_result():
    calibration = StructuralValidationCalibration(
        experiment_seed=123,
        n_agents=8,
        k=2,
        n_replications=2,
        hub_q=2,
        p_sw=0.1,
        a0=1.0,
    )
    return run_structural_ensemble(
        experiment_seed=calibration.experiment_seed,
        n_replications=calibration.n_replications,
        n_agents=calibration.n_agents,
        q=calibration.hub_q,
        specifications=calibration.topology_specifications(),
    )


def test_first_structural_validation_calibration_matches_decision_d041() -> None:
    calibration = first_structural_validation_calibration()
    assert calibration.experiment_seed == 20260901
    assert calibration.n_agents == 100
    assert calibration.k == 6
    assert calibration.n_replications == 1000
    assert calibration.hub_q == 5
    assert calibration.p_sw == 0.02
    assert calibration.a0 == 1.0


def test_calibration_builds_expected_matched_topology_specifications() -> None:
    specifications = first_structural_validation_calibration().topology_specifications()
    assert tuple(spec.topology_label for spec in specifications) == ("R", "SW", "SF")
    assert tuple(spec.kind for spec in specifications) == (
        "random",
        "small_world",
        "hub_dominated",
    )
    assert {spec.k for spec in specifications} == {6}
    assert specifications[1].p_sw == 0.02
    assert specifications[2].a0 == 1.0


def test_calibration_rejects_odd_k() -> None:
    kwargs = calibration_kwargs()
    kwargs["k"] = 5
    with pytest.raises(ValueError, match="even"):
        StructuralValidationCalibration(**kwargs)


def test_calibration_rejects_hub_q_above_population() -> None:
    kwargs = calibration_kwargs()
    kwargs["hub_q"] = 101
    with pytest.raises(ValueError, match="hub_q"):
        StructuralValidationCalibration(**kwargs)


def test_calibration_rejects_invalid_small_world_probability() -> None:
    kwargs = calibration_kwargs()
    kwargs["p_sw"] = 1.1
    with pytest.raises(ValueError, match="p_sw"):
        StructuralValidationCalibration(**kwargs)


def test_calibration_rejects_nonpositive_initial_attractiveness() -> None:
    kwargs = calibration_kwargs()
    kwargs["a0"] = 0.0
    with pytest.raises(ValueError, match="a0"):
        StructuralValidationCalibration(**kwargs)


def test_write_structural_result_creates_all_three_artifacts(tmp_path: Path) -> None:
    paths = write_structural_result(tiny_result(), output_directory=tmp_path)
    assert len(paths) == 3
    assert all(path.exists() for path in paths)
    assert {path.name for path in paths} == {
        "structural_graph_records.csv",
        "structural_summary.csv",
        "structural_metadata.json",
    }


def test_raw_structural_csv_preserves_every_graph_level_record(tmp_path: Path) -> None:
    result = tiny_result()
    raw_path, _, _ = write_structural_result(result, output_directory=tmp_path)
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.records) == 6


def test_raw_structural_csv_contains_report_defined_metrics(tmp_path: Path) -> None:
    raw_path, _, _ = write_structural_result(tiny_result(), output_directory=tmp_path)
    with raw_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    assert fieldnames is not None
    assert {
        "in_degree_gini",
        "hub_link_share",
        "global_clustering",
        "average_path_length_lcc",
        "largest_component_share",
    }.issubset(set(fieldnames))


def test_summary_csv_contains_five_metrics_per_topology(tmp_path: Path) -> None:
    _, summary_path, _ = write_structural_result(tiny_result(), output_directory=tmp_path)
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert {row["topology_label"] for row in rows} == {"R", "SW", "SF"}


def test_metadata_json_records_structural_design(tmp_path: Path) -> None:
    result = tiny_result()
    _, _, metadata_path = write_structural_result(result, output_directory=tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["experiment_seed"] == 123
    assert metadata["n_agents"] == 8
    assert metadata["n_replications"] == 2
    assert metadata["hub_q"] == 2
    assert metadata["n_records"] == 6
    assert [item["topology_label"] for item in metadata["topologies"]] == ["R", "SW", "SF"]


def test_write_structural_result_rejects_wrong_result_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="StructuralEnsembleResult"):
        write_structural_result("not-a-result", output_directory=tmp_path)


def test_structural_validation_cli_smoke_run_succeeds(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_refined_structural_validation.py",
            "--n-replications",
            "2",
            "--outdir",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "structural_graph_records.csv").exists()
    assert (tmp_path / "structural_summary.csv").exists()
    assert (tmp_path / "structural_metadata.json").exists()


def test_structural_validation_cli_reports_all_three_topology_labels(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_refined_structural_validation.py",
            "--n-replications",
            "2",
            "--outdir",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "R" in completed.stdout
    assert "SW" in completed.stdout
    assert "SF" in completed.stdout
