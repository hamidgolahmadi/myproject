"""Thesis-ready plotting from finalized D049 joint-interaction artifacts.

This module is downstream-only: it reads the finalized D049 analysis JSON and
never recomputes simulations, bootstrap resamples, or economic transitions.
Point estimates and matched-block cellwise SF-SW intervals come from the final
analysis artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALPHA_LEVELS = (0.40, 0.85, 0.99)
BETA_LEVELS = (0.0, 1.0, 5.0, 100.0)
GAMMA_LEVELS = (0.0, 0.90, 0.99, 0.999)
TOPOLOGIES = ("R", "SW", "SF")

SURFACE_OUTCOMES = (
    "mean_attention_overlap",
    "mean_pairwise_action_covariance",
    "mean_aggregate_order_flow_variance",
    "return_volatility",
)

OUTCOME_LABELS = {
    "mean_attention_overlap": "Attention overlap",
    "mean_pairwise_action_covariance": "Pairwise action covariance",
    "mean_aggregate_order_flow_variance": "Aggregate order-flow variance",
    "return_volatility": "Return volatility",
}

BETA_MARKERS = {
    0.0: "o",
    1.0: "s",
    5.0: "^",
    100.0: "D",
}


@dataclass(frozen=True, slots=True)
class JointInteractionPlotData:
    """Validated finalized D049 data needed for thesis figures."""

    n_replications: int
    confidence_level: float
    topology_means: tuple[dict, ...]
    pairwise_contrasts: tuple[dict, ...]


def load_joint_interaction_plot_data(results_dir: str | Path) -> JointInteractionPlotData:
    """Load and validate the finalized D049 analysis artifact."""

    results_dir = Path(results_dir)
    path = results_dir / "joint_analysis.json"
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        analysis = json.load(handle)

    if int(analysis.get("n_cells", 0)) != 50:
        raise ValueError("D049 plotting requires the finalized 50-cell design")
    if analysis.get("cross_cell_common_random_numbers_verified") is not True:
        raise ValueError("D049 plotting requires verified cross-cell CRN")
    if analysis.get("alpha_zero_economic_path_null_verified") is not True:
        raise ValueError("D049 plotting requires the verified alpha-zero null")

    n_replications = int(analysis.get("n_replications", 0))
    confidence_level = float(analysis.get("confidence_level", 0.0))
    if n_replications <= 0:
        raise ValueError("D049 plotting requires a positive replication count")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("D049 plotting requires a valid confidence level")

    topology_means = tuple(analysis.get("topology_means", ()))
    pairwise_contrasts = tuple(analysis.get("pairwise_contrasts", ()))
    if not topology_means or not pairwise_contrasts:
        raise ValueError("D049 analysis artifact is missing final mean/contrast rows")

    factorial_means = [row for row in topology_means if row.get("cell_kind") == "factorial"]
    observed_cells = {
        (float(row["alpha"]), float(row["beta"]), float(row["gamma_R"]))
        for row in factorial_means
    }
    expected_cells = {
        (alpha, beta, gamma)
        for alpha in ALPHA_LEVELS
        for beta in BETA_LEVELS
        for gamma in GAMMA_LEVELS
    }
    if observed_cells != expected_cells:
        raise ValueError("D049 plotting inputs do not cover the exact frozen factorial grid")

    return JointInteractionPlotData(
        n_replications=n_replications,
        confidence_level=confidence_level,
        topology_means=topology_means,
        pairwise_contrasts=pairwise_contrasts,
    )


def _match_float(left: object, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))


def _factorial_mean(
    data: JointInteractionPlotData,
    *,
    alpha: float,
    beta: float,
    gamma_R: float,
    outcome: str,
    topology: str,
) -> float:
    rows = [
        row
        for row in data.topology_means
        if row.get("cell_kind") == "factorial"
        and _match_float(row["alpha"], alpha)
        and _match_float(row["beta"], beta)
        and _match_float(row["gamma_R"], gamma_R)
        and row["outcome"] == outcome
        and row["topology"] == topology
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one D049 topology mean for {(alpha, beta, gamma_R, outcome, topology)}"
        )
    return float(rows[0]["estimate"])


def signed_sf_minus_sw(
    data: JointInteractionPlotData,
    *,
    alpha: float,
    beta: float,
    gamma_R: float,
    outcome: str,
) -> tuple[float, float, float]:
    """Return final D=SF-SW estimate and matched-block 95% interval."""

    rows = [
        row
        for row in data.pairwise_contrasts
        if row.get("cell_kind") == "factorial"
        and _match_float(row["alpha"], alpha)
        and _match_float(row["beta"], beta)
        and _match_float(row["gamma_R"], gamma_R)
        and row["outcome"] == outcome
        and row["topology_left"] == "SW"
        and row["topology_right"] == "SF"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one D049 SW-SF contrast for {(alpha, beta, gamma_R, outcome)}"
        )
    row = rows[0]
    sw_minus_sf = float(row["estimate"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    return -sw_minus_sf, -upper, -lower


def _save_figure(
    fig,
    *,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        suffix = suffix.lower().lstrip(".")
        if suffix not in {"png", "pdf"}:
            raise ValueError("figure formats are limited to png and pdf")
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return tuple(paths)


def _discrete_axis(ax, values: Sequence[float], *, label: str) -> np.ndarray:
    x = np.arange(len(values), dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in values])
    ax.set_xlabel(label)
    return x


def plot_beta_gamma_surface(
    data: JointInteractionPlotData,
    *,
    outcome: str,
    alpha: float = 0.85,
    output_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot D=SF-SW over gamma_R for each beta at one alpha."""

    if outcome not in SURFACE_OUTCOMES:
        raise ValueError(f"unsupported D049 surface outcome: {outcome}")
    if alpha not in ALPHA_LEVELS:
        raise ValueError("alpha must be one of the frozen factorial levels")

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    x = _discrete_axis(ax, GAMMA_LEVELS, label=r"Reputation persistence, $\gamma_R$")

    for beta in BETA_LEVELS:
        estimates: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for gamma_R in GAMMA_LEVELS:
            estimate, lower, upper = signed_sf_minus_sw(
                data,
                alpha=alpha,
                beta=beta,
                gamma_R=gamma_R,
                outcome=outcome,
            )
            estimates.append(estimate)
            lowers.append(lower)
            uppers.append(upper)

        estimate_arr = np.asarray(estimates)
        lower_arr = np.asarray(lowers)
        upper_arr = np.asarray(uppers)
        yerr = np.vstack((estimate_arr - lower_arr, upper_arr - estimate_arr))
        ax.errorbar(
            x,
            estimate_arr,
            yerr=yerr,
            marker=BETA_MARKERS[beta],
            linewidth=1.5,
            capsize=2.4,
            label=rf"$\beta={beta:g}$",
        )

    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel(rf"SF-SW: {OUTCOME_LABELS[outcome]}")
    ax.set_title(rf"D049 joint surface at $\alpha={alpha:g}$")
    ax.legend(title=r"Selectivity $\beta$", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    safe_outcome = outcome.replace("mean_", "")
    return _save_figure(
        fig,
        output_dir=Path(output_dir),
        stem=f"d049_beta_gamma_{safe_outcome}_alpha085",
        formats=formats,
        dpi=dpi,
    )


def plot_alpha_boundary_return_volatility(
    data: JointInteractionPlotData,
    *,
    beta: float = 5.0,
    gamma_R: float = 0.90,
    output_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot the high-alpha SF-SW return-volatility ranking reversal."""

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    x = _discrete_axis(ax, ALPHA_LEVELS, label=r"Social transmission weight, $\alpha$")
    estimates: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for alpha in ALPHA_LEVELS:
        estimate, lower, upper = signed_sf_minus_sw(
            data,
            alpha=alpha,
            beta=beta,
            gamma_R=gamma_R,
            outcome="return_volatility",
        )
        estimates.append(estimate)
        lowers.append(lower)
        uppers.append(upper)

    estimate_arr = np.asarray(estimates)
    lower_arr = np.asarray(lowers)
    upper_arr = np.asarray(uppers)
    yerr = np.vstack((estimate_arr - lower_arr, upper_arr - estimate_arr))
    ax.errorbar(x, estimate_arr, yerr=yerr, marker="o", linewidth=1.6, capsize=2.6)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel("SF-SW return volatility")
    ax.set_title(rf"High-$\alpha$ boundary at $\beta={beta:g},\ \gamma_R={gamma_R:g}$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return _save_figure(
        fig,
        output_dir=Path(output_dir),
        stem="d049_alpha_boundary_return_volatility_beta5_gamma09",
        formats=formats,
        dpi=dpi,
    )


def plot_sigma0_boundary_diagnostic(
    data: JointInteractionPlotData,
    *,
    alpha: float = 0.85,
    beta: float = 5.0,
    output_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot pooled raw-reputation dispersion relative to the fixed sigma_0 floor."""

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    x = _discrete_axis(ax, GAMMA_LEVELS, label=r"Reputation persistence, $\gamma_R$")
    pooled: list[float] = []
    for gamma_R in GAMMA_LEVELS:
        values = [
            _factorial_mean(
                data,
                alpha=alpha,
                beta=beta,
                gamma_R=gamma_R,
                outcome="mean_raw_local_reputation_std_over_sigma0",
                topology=topology,
            )
            for topology in TOPOLOGIES
        ]
        pooled.append(float(np.mean(values)))

    ax.plot(x, pooled, marker="o", linewidth=1.7)
    ax.axhline(1.0, linewidth=1.0, linestyle="--", label=r"raw std = $\sigma_0$")
    ax.set_ylabel(r"Pooled raw reputation std / $\sigma_0$")
    ax.set_title(rf"Regularisation-floor diagnostic at $\alpha={alpha:g},\ \beta={beta:g}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return _save_figure(
        fig,
        output_dir=Path(output_dir),
        stem="d049_sigma0_boundary_diagnostic_alpha085_beta5",
        formats=formats,
        dpi=dpi,
    )


def generate_joint_interaction_figures(
    *,
    results_dir: str | Path,
    output_dir: str | Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Generate the six main D049 thesis figures from finalized artifacts."""

    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats):
        raise ValueError("figure formats must be non-empty and unique")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    data = load_joint_interaction_plot_data(results_dir)
    paths: list[Path] = []
    for outcome in SURFACE_OUTCOMES:
        paths.extend(
            plot_beta_gamma_surface(
                data,
                outcome=outcome,
                alpha=0.85,
                output_dir=output_dir,
                formats=formats,
                dpi=dpi,
            )
        )
    paths.extend(
        plot_alpha_boundary_return_volatility(
            data,
            output_dir=output_dir,
            formats=formats,
            dpi=dpi,
        )
    )
    paths.extend(
        plot_sigma0_boundary_diagnostic(
            data,
            output_dir=output_dir,
            formats=formats,
            dpi=dpi,
        )
    )
    return tuple(paths)
