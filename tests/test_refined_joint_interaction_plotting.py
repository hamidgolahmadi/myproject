import json
from pathlib import Path

import pytest

from src.experiments.refined.joint_interaction_plotting import (
    ALPHA_LEVELS,
    BETA_LEVELS,
    GAMMA_LEVELS,
    SURFACE_OUTCOMES,
    generate_joint_interaction_figures,
    load_joint_interaction_plot_data,
    signed_sf_minus_sw,
)


TOPOLOGIES = ("R", "SW", "SF")


def _write_finalized_fixture(root: Path, *, crn_verified: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    means = []
    contrasts = []

    for a_index, alpha in enumerate(ALPHA_LEVELS):
        for b_index, beta in enumerate(BETA_LEVELS):
            for g_index, gamma in enumerate(GAMMA_LEVELS):
                cell_id = f"F{a_index}{b_index}{g_index}"
                for t_index, topology in enumerate(TOPOLOGIES):
                    ratio = 3.0 - 0.5 * g_index + 0.01 * t_index
                    means.append(
                        {
                            "family": "mechanism",
                            "cell_id": cell_id,
                            "cell_kind": "factorial",
                            "alpha": alpha,
                            "beta": beta,
                            "gamma_R": gamma,
                            "outcome": "mean_raw_local_reputation_std_over_sigma0",
                            "topology": topology,
                            "estimate": ratio,
                            "ci_lower": ratio - 0.1,
                            "ci_upper": ratio + 0.1,
                            "n_replications": 3,
                        }
                    )

                for outcome_index, outcome in enumerate(SURFACE_OUTCOMES):
                    d = (
                        0.001 * (outcome_index + 1)
                        + 0.0001 * a_index
                        + 0.00001 * b_index
                        + 0.000001 * g_index
                    )
                    contrasts.append(
                        {
                            "family": "mechanism",
                            "cell_id": cell_id,
                            "cell_kind": "factorial",
                            "alpha": alpha,
                            "beta": beta,
                            "gamma_R": gamma,
                            "outcome": outcome,
                            "topology_left": "SW",
                            "topology_right": "SF",
                            "estimate": -d,
                            "ci_lower": -(d + 0.0002),
                            "ci_upper": -(d - 0.0002),
                            "relative_effect": None,
                            "relative_ci_lower": None,
                            "relative_ci_upper": None,
                            "n_replications": 3,
                        }
                    )

    payload = {
        "experiment_seed": 2026091003,
        "bootstrap_seed": 2026091004,
        "n_replications": 3,
        "n_bootstrap": 5000,
        "confidence_level": 0.95,
        "n_cells": 50,
        "cross_cell_common_random_numbers_verified": crn_verified,
        "alpha_zero_economic_path_null_verified": True,
        "topology_means": means,
        "topology_gaps": [],
        "pairwise_contrasts": contrasts,
        "interactions": [],
    }
    (root / "joint_analysis.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_load_joint_interaction_plot_data_validates_final_design(tmp_path):
    root = _write_finalized_fixture(tmp_path / "joint")
    data = load_joint_interaction_plot_data(root)
    assert data.n_replications == 3
    assert data.confidence_level == pytest.approx(0.95)
    assert len(data.topology_means) == 3 * 4 * 4 * 3


def test_signed_sf_minus_sw_inverts_final_sw_sf_contrast(tmp_path):
    root = _write_finalized_fixture(tmp_path / "joint")
    data = load_joint_interaction_plot_data(root)
    estimate, lower, upper = signed_sf_minus_sw(
        data,
        alpha=0.40,
        beta=0.0,
        gamma_R=0.0,
        outcome="mean_attention_overlap",
    )
    assert estimate == pytest.approx(0.001)
    assert lower == pytest.approx(0.0008)
    assert upper == pytest.approx(0.0012)


def test_generate_joint_interaction_figures_writes_six_main_outputs(tmp_path):
    root = _write_finalized_fixture(tmp_path / "joint")
    output = tmp_path / "figures"
    paths = generate_joint_interaction_figures(
        results_dir=root,
        output_dir=output,
        formats=("png",),
    )
    assert len(paths) == 6
    assert {path.name for path in paths} == {
        "d049_beta_gamma_attention_overlap_alpha085.png",
        "d049_beta_gamma_pairwise_action_covariance_alpha085.png",
        "d049_beta_gamma_aggregate_order_flow_variance_alpha085.png",
        "d049_beta_gamma_return_volatility_alpha085.png",
        "d049_alpha_boundary_return_volatility_beta5_gamma09.png",
        "d049_sigma0_boundary_diagnostic_alpha085_beta5.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_generate_joint_interaction_figures_writes_png_and_pdf(tmp_path):
    root = _write_finalized_fixture(tmp_path / "joint")
    paths = generate_joint_interaction_figures(
        results_dir=root,
        output_dir=tmp_path / "figures",
        formats=("png", "pdf"),
    )
    assert len(paths) == 12
    assert sum(path.suffix == ".png" for path in paths) == 6
    assert sum(path.suffix == ".pdf" for path in paths) == 6


def test_load_joint_interaction_plot_data_rejects_unverified_crn(tmp_path):
    root = _write_finalized_fixture(tmp_path / "joint", crn_verified=False)
    with pytest.raises(ValueError, match="verified cross-cell CRN"):
        load_joint_interaction_plot_data(root)


def test_generate_joint_interaction_figures_rejects_duplicate_formats(tmp_path):
    root = _write_finalized_fixture(tmp_path / "joint")
    with pytest.raises(ValueError, match="non-empty and unique"):
        generate_joint_interaction_figures(
            results_dir=root,
            output_dir=tmp_path / "figures",
            formats=("png", "png"),
        )
