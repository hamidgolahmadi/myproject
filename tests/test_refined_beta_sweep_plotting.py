import csv
import json
from pathlib import Path

import pytest

from src.experiments.refined.beta_sweep_plotting import (
    generate_beta_sweep_figures,
    load_beta_sweep_plot_data,
    x_coordinates,
)


BETAS = (0.0, 0.1, 1.0, 10.0)
RELATIVE_OUTCOMES = (
    "return_volatility",
    "mean_absolute_order_flow_per_agent",
    "peak_cid",
    "mean_aggregate_order_flow_variance",
    "mean_hub_influence_share",
    "mean_attention_overlap",
)
ACTION_COVARIANCE = "mean_pairwise_action_covariance"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_finalized_fixture(root: Path, *, confirmatory: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "final_beta_sweep": True,
        "confirmatory": confirmatory,
        "n_complete_replications_per_beta": 3,
        "protocol": {
            "beta_grid": list(BETAS),
            "confidence_level": 0.95,
        },
    }
    (root / "beta_sweep_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    means: list[dict[str, object]] = []
    for beta_index, beta in enumerate(BETAS):
        for topology_index, topology in enumerate(("R", "SW", "SF")):
            estimate = 1.0 + 0.01 * beta_index + 0.001 * topology_index
            means.append(
                {
                    "family": "primary",
                    "beta": beta,
                    "outcome": "peak_cid",
                    "topology": topology,
                    "estimate": estimate,
                    "ci_lower": estimate - 0.01,
                    "ci_upper": estimate + 0.01,
                    "n_replications": 3,
                }
            )
    _write_csv(root / "beta_topology_means.csv", means)

    gaps: list[dict[str, object]] = []
    for beta_index, beta in enumerate(BETAS):
        for outcome_index, outcome in enumerate((*RELATIVE_OUTCOMES, ACTION_COVARIANCE)):
            absolute = 0.01 * (1 + beta_index + outcome_index)
            relative = 0.02 * (1 + beta_index + outcome_index)
            has_relative = outcome != ACTION_COVARIANCE
            gaps.append(
                {
                    "family": "mechanism",
                    "beta": beta,
                    "outcome": outcome,
                    "absolute_gap": absolute,
                    "absolute_gap_ci_lower": absolute * 0.9,
                    "absolute_gap_ci_upper": absolute * 1.1,
                    "relative_gap": relative if has_relative else "",
                    "relative_gap_ci_lower": relative * 0.9 if has_relative else "",
                    "relative_gap_ci_upper": relative * 1.1 if has_relative else "",
                    "n_replications": 3,
                }
            )
    _write_csv(root / "beta_topology_gaps.csv", gaps)
    return root


def test_load_beta_sweep_plot_data_uses_finalized_grid(tmp_path):
    root = _write_finalized_fixture(tmp_path / "beta")
    data = load_beta_sweep_plot_data(root)
    assert data.beta_grid == BETAS
    assert data.n_replications == 3
    assert data.confidence_level == pytest.approx(0.95)


def test_x_coordinates_grid_keeps_zero_and_log_excludes_zero():
    grid_x, grid_indices = x_coordinates(BETAS, mode="grid")
    log_x, log_indices = x_coordinates(BETAS, mode="log")
    assert tuple(grid_indices) == (0, 1, 2, 3)
    assert tuple(grid_x) == pytest.approx((0.0, 1.0, 2.0, 3.0))
    assert tuple(log_indices) == (1, 2, 3)
    assert tuple(log_x) == pytest.approx((0.1, 1.0, 10.0))


def test_x_coordinates_symlog_keeps_zero_and_actual_beta_values():
    symlog_x, symlog_indices = x_coordinates(BETAS, mode="symlog")
    assert tuple(symlog_indices) == (0, 1, 2, 3)
    assert tuple(symlog_x) == pytest.approx(BETAS)


def test_generate_beta_sweep_figures_writes_split_main_grid_outputs(tmp_path):
    root = _write_finalized_fixture(tmp_path / "beta")
    output = tmp_path / "figures"
    paths = generate_beta_sweep_figures(
        results_dir=root,
        output_dir=output,
        modes=("grid",),
        formats=("png",),
    )
    assert len(paths) == 5
    assert {path.name for path in paths} == {
        "d047_peak_cid_levels_grid.png",
        "d047_market_core_relative_gap_summary_grid.png",
        "d047_aggregate_flow_variance_relative_gap_grid.png",
        "d047_mechanism_relative_gap_summary_grid.png",
        "d047_action_covariance_absolute_gap_grid.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_generate_beta_sweep_figures_defaults_to_symlog_thesis_set(tmp_path):
    root = _write_finalized_fixture(tmp_path / "beta")
    output = tmp_path / "figures"
    paths = generate_beta_sweep_figures(
        results_dir=root,
        output_dir=output,
        formats=("png",),
    )
    assert len(paths) == 5
    assert all(path.name.endswith("_symlog.png") for path in paths)
    assert (output / "d047_aggregate_flow_variance_relative_gap_symlog.png").exists()


def test_generate_beta_sweep_figures_detail_adds_five_relative_gap_plots(tmp_path):
    root = _write_finalized_fixture(tmp_path / "beta")
    paths = generate_beta_sweep_figures(
        results_dir=root,
        output_dir=tmp_path / "figures",
        modes=("grid",),
        formats=("png",),
        include_detail=True,
    )
    assert len(paths) == 10
    assert sum("relative_gap_grid" in path.name for path in paths) >= 5


def test_generate_beta_sweep_figures_rejects_duplicate_modes(tmp_path):
    root = _write_finalized_fixture(tmp_path / "beta")
    with pytest.raises(ValueError, match="unique plotting modes"):
        generate_beta_sweep_figures(
            results_dir=root,
            output_dir=tmp_path / "figures",
            modes=("symlog", "symlog"),
            formats=("png",),
        )


def test_load_beta_sweep_plot_data_rejects_confirmatory_metadata(tmp_path):
    root = _write_finalized_fixture(tmp_path / "beta", confirmatory=True)
    with pytest.raises(ValueError, match="confirmatory=False"):
        load_beta_sweep_plot_data(root)
