"""Reproducible figures for the completed exploratory D047 beta sweep.

The plotting layer is intentionally downstream of the frozen D047 analysis.  It
reads only the final CSV/JSON artifacts and never recomputes simulation outcomes,
bootstrap intervals, topology gaps, or any economic transition.

Two x-axis representations are supported:

``grid``
    The predeclared beta values are shown at equally spaced positions, including
    beta=0.  This is the main full-design display because a literal linear axis
    would compress all beta <= 10 near zero when beta_max=1000.

``log``
    The actual positive beta values are shown on a logarithmic x axis.  beta=0
    is omitted because log(0) is undefined.  This view is useful for resolving
    the transition and high-selectivity regions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOPOLOGY_ORDER = ("R", "SW", "SF")
TOPOLOGY_MARKERS = {"R": "o", "SW": "s", "SF": "^"}

OUTCOME_LABELS = {
    "return_volatility": "Return-volatility topology gap",
    "mean_absolute_order_flow_per_agent": "Mean absolute order-flow topology gap",
    "peak_cid": "Peak-CID topology gap",
    "mean_aggregate_order_flow_variance": "Aggregate order-flow variance topology gap",
    "mean_hub_influence_share": "Structural-hub influence topology gap",
    "mean_attention_overlap": "Attention-overlap topology gap",
    "mean_pairwise_action_covariance": "Pairwise action-covariance topology gap",
}

MARKET_RELATIVE_GAP_OUTCOMES = (
    "return_volatility",
    "mean_absolute_order_flow_per_agent",
    "peak_cid",
    "mean_aggregate_order_flow_variance",
)

MECHANISM_RELATIVE_GAP_OUTCOMES = (
    "mean_hub_influence_share",
    "mean_attention_overlap",
)

ACTION_COVARIANCE_OUTCOME = "mean_pairwise_action_covariance"


@dataclass(frozen=True, slots=True)
class BetaSweepPlotData:
    """Validated plotting inputs loaded from final D047 artifacts."""

    beta_grid: tuple[float, ...]
    means: tuple[dict[str, str], ...]
    gaps: tuple[dict[str, str], ...]
    n_replications: int
    confidence_level: float


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"plot input is empty: {path}")
    return rows


def _require_columns(rows: Sequence[dict[str, str]], required: set[str], label: str) -> None:
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_beta_sweep_plot_data(results_dir: str | Path) -> BetaSweepPlotData:
    """Load and validate the finalized D047 mean/gap artifacts."""

    results_dir = Path(results_dir)
    metadata_path = results_dir / "beta_sweep_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    if metadata.get("final_beta_sweep") is not True:
        raise ValueError("D047 plotting requires final_beta_sweep=True metadata")
    if metadata.get("confirmatory") is not False:
        raise ValueError("D047 plotting expects exploratory confirmatory=False metadata")

    protocol = metadata.get("protocol", {})
    beta_grid = tuple(float(value) for value in protocol.get("beta_grid", ()))
    if not beta_grid:
        raise ValueError("D047 metadata does not contain a beta grid")
    if beta_grid[0] != 0.0 or any(beta < 0.0 for beta in beta_grid):
        raise ValueError("D047 plotting expects a non-negative beta grid beginning at zero")
    if tuple(sorted(beta_grid)) != beta_grid or len(set(beta_grid)) != len(beta_grid):
        raise ValueError("D047 beta grid must be strictly increasing")

    means = _read_csv(results_dir / "beta_topology_means.csv")
    gaps = _read_csv(results_dir / "beta_topology_gaps.csv")
    _require_columns(
        means,
        {"beta", "outcome", "topology", "estimate", "ci_lower", "ci_upper"},
        "beta_topology_means.csv",
    )
    _require_columns(
        gaps,
        {
            "beta",
            "outcome",
            "absolute_gap",
            "absolute_gap_ci_lower",
            "absolute_gap_ci_upper",
            "relative_gap",
            "relative_gap_ci_lower",
            "relative_gap_ci_upper",
        },
        "beta_topology_gaps.csv",
    )

    observed_mean_betas = {float(row["beta"]) for row in means}
    observed_gap_betas = {float(row["beta"]) for row in gaps}
    expected_betas = set(beta_grid)
    if observed_mean_betas != expected_betas or observed_gap_betas != expected_betas:
        raise ValueError("D047 plotting inputs do not cover the exact frozen beta grid")

    n_replications = int(metadata.get("n_complete_replications_per_beta", 0))
    confidence_level = float(protocol.get("confidence_level", 0.0))
    if n_replications <= 0:
        raise ValueError("D047 metadata has an invalid replication count")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("D047 metadata has an invalid confidence level")

    return BetaSweepPlotData(
        beta_grid=beta_grid,
        means=means,
        gaps=gaps,
        n_replications=n_replications,
        confidence_level=confidence_level,
    )


def x_coordinates(
    beta_grid: Sequence[float],
    *,
    mode: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return plotting x coordinates and selected beta-grid indices."""

    if mode == "grid":
        indices = tuple(range(len(beta_grid)))
        return np.arange(len(beta_grid), dtype=float), indices
    if mode == "log":
        indices = tuple(i for i, beta in enumerate(beta_grid) if float(beta) > 0.0)
        if not indices:
            raise ValueError("log-beta display requires at least one strictly positive beta")
        return np.asarray([float(beta_grid[i]) for i in indices], dtype=float), indices
    raise ValueError("mode must be 'grid' or 'log'")


def _beta_label(beta: float) -> str:
    return f"{beta:g}"


def _configure_x_axis(ax, beta_grid: Sequence[float], *, mode: str) -> tuple[np.ndarray, tuple[int, ...]]:
    x, indices = x_coordinates(beta_grid, mode=mode)
    selected = [float(beta_grid[i]) for i in indices]
    if mode == "grid":
        ax.set_xticks(x)
        ax.set_xticklabels([_beta_label(beta) for beta in selected])
        ax.set_xlabel(r"$\beta$ (predeclared grid; equally spaced display)")
    else:
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([_beta_label(beta) for beta in selected])
        ax.set_xlabel(r"$\beta$ (log scale; $\beta>0$)")
    return x, indices


def _rows_for_outcome(rows: Iterable[dict[str, str]], outcome: str) -> tuple[dict[str, str], ...]:
    selected = tuple(row for row in rows if row["outcome"] == outcome)
    if not selected:
        raise ValueError(f"D047 plotting outcome not found: {outcome}")
    return selected


def _ordered_by_beta(
    rows: Iterable[dict[str, str]], beta_grid: Sequence[float]
) -> tuple[dict[str, str], ...]:
    by_beta = {float(row["beta"]): row for row in rows}
    missing = [beta for beta in beta_grid if float(beta) not in by_beta]
    if missing:
        raise ValueError(f"plot rows are missing beta values: {missing}")
    return tuple(by_beta[float(beta)] for beta in beta_grid)


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


def plot_peak_cid_topology_levels(
    data: BetaSweepPlotData,
    *,
    output_dir: str | Path,
    mode: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot Peak CID topology means with matched-block 95% intervals."""

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x, indices = _configure_x_axis(ax, data.beta_grid, mode=mode)
    rows = _rows_for_outcome(data.means, "peak_cid")

    for topology in TOPOLOGY_ORDER:
        topology_rows = _ordered_by_beta(
            (row for row in rows if row["topology"] == topology), data.beta_grid
        )
        estimate = np.asarray([float(row["estimate"]) for row in topology_rows])[list(indices)]
        lower = np.asarray([float(row["ci_lower"]) for row in topology_rows])[list(indices)]
        upper = np.asarray([float(row["ci_upper"]) for row in topology_rows])[list(indices)]
        yerr = np.vstack((estimate - lower, upper - estimate))
        ax.errorbar(
            x,
            estimate,
            yerr=yerr,
            marker=TOPOLOGY_MARKERS[topology],
            linewidth=1.5,
            capsize=2.5,
            label=topology,
        )

    ax.set_ylabel("Peak CID")
    ax.set_title("D047: Peak CID across reputational selectivity")
    ax.legend(title="Topology")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_figure(
        fig,
        output_dir=Path(output_dir),
        stem=f"d047_peak_cid_levels_{mode}",
        formats=formats,
        dpi=dpi,
    )


def _relative_gap_series(
    data: BetaSweepPlotData, outcome: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _ordered_by_beta(_rows_for_outcome(data.gaps, outcome), data.beta_grid)
    try:
        estimate = np.asarray([100.0 * float(row["relative_gap"]) for row in rows])
        lower = np.asarray([100.0 * float(row["relative_gap_ci_lower"]) for row in rows])
        upper = np.asarray([100.0 * float(row["relative_gap_ci_upper"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"outcome {outcome!r} does not have D047 relative-gap values") from exc
    return estimate, lower, upper


def _absolute_gap_series(
    data: BetaSweepPlotData, outcome: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _ordered_by_beta(_rows_for_outcome(data.gaps, outcome), data.beta_grid)
    estimate = np.asarray([float(row["absolute_gap"]) for row in rows])
    lower = np.asarray([float(row["absolute_gap_ci_lower"]) for row in rows])
    upper = np.asarray([float(row["absolute_gap_ci_upper"]) for row in rows])
    return estimate, lower, upper


def plot_relative_gap_summary(
    data: BetaSweepPlotData,
    *,
    outcomes: Sequence[str],
    title: str,
    stem: str,
    output_dir: str | Path,
    mode: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot point-estimate relative topology gaps for a comparable metric family."""

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    x, indices = _configure_x_axis(ax, data.beta_grid, mode=mode)
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for marker, outcome in zip(markers, outcomes, strict=False):
        estimate, _, _ = _relative_gap_series(data, outcome)
        ax.plot(
            x,
            estimate[list(indices)],
            marker=marker,
            linewidth=1.6,
            label=OUTCOME_LABELS[outcome],
        )

    ax.set_ylabel("Relative topology gap (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_figure(
        fig,
        output_dir=Path(output_dir),
        stem=f"{stem}_{mode}",
        formats=formats,
        dpi=dpi,
    )


def plot_relative_gap_detail(
    data: BetaSweepPlotData,
    *,
    outcome: str,
    output_dir: str | Path,
    mode: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot one relative topology gap with its matched-block percentile interval."""

    estimate, lower, upper = _relative_gap_series(data, outcome)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x, indices = _configure_x_axis(ax, data.beta_grid, mode=mode)
    estimate = estimate[list(indices)]
    lower = lower[list(indices)]
    upper = upper[list(indices)]
    yerr = np.vstack((estimate - lower, upper - estimate))
    ax.errorbar(x, estimate, yerr=yerr, marker="o", linewidth=1.6, capsize=2.5)
    ax.set_ylabel("Relative topology gap (%)")
    ax.set_title(f"D047: {OUTCOME_LABELS[outcome]}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_figure(
        fig,
        output_dir=Path(output_dir),
        stem=f"d047_{outcome}_relative_gap_{mode}",
        formats=formats,
        dpi=dpi,
    )


def plot_action_covariance_absolute_gap(
    data: BetaSweepPlotData,
    *,
    output_dir: str | Path,
    mode: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot the precomputed absolute action-covariance topology gap with CI."""

    estimate, lower, upper = _absolute_gap_series(data, ACTION_COVARIANCE_OUTCOME)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x, indices = _configure_x_axis(ax, data.beta_grid, mode=mode)
    estimate = estimate[list(indices)]
    lower = lower[list(indices)]
    upper = upper[list(indices)]
    yerr = np.vstack((estimate - lower, upper - estimate))
    ax.errorbar(x, estimate, yerr=yerr, marker="o", linewidth=1.6, capsize=2.5)
    ax.set_ylabel("Absolute topology gap")
    ax.set_title("D047: Pairwise action-covariance topology gap")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _save_figure(
        fig,
        output_dir=Path(output_dir),
        stem=f"d047_action_covariance_absolute_gap_{mode}",
        formats=formats,
        dpi=dpi,
    )


def generate_beta_sweep_figures(
    *,
    results_dir: str | Path,
    output_dir: str | Path,
    modes: Sequence[str] = ("grid", "log"),
    formats: Sequence[str] = ("png", "pdf"),
    include_detail: bool = False,
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Generate the main D047 thesis figures from finalized artifacts."""

    data = load_beta_sweep_plot_data(results_dir)
    paths: list[Path] = []
    for mode in modes:
        paths.extend(
            plot_peak_cid_topology_levels(
                data,
                output_dir=output_dir,
                mode=mode,
                formats=formats,
                dpi=dpi,
            )
        )
        paths.extend(
            plot_relative_gap_summary(
                data,
                outcomes=MARKET_RELATIVE_GAP_OUTCOMES,
                title="D047: Market topology gaps across reputational selectivity",
                stem="d047_market_relative_gap_summary",
                output_dir=output_dir,
                mode=mode,
                formats=formats,
                dpi=dpi,
            )
        )
        paths.extend(
            plot_relative_gap_summary(
                data,
                outcomes=MECHANISM_RELATIVE_GAP_OUTCOMES,
                title="D047: Mechanism topology gaps across reputational selectivity",
                stem="d047_mechanism_relative_gap_summary",
                output_dir=output_dir,
                mode=mode,
                formats=formats,
                dpi=dpi,
            )
        )
        paths.extend(
            plot_action_covariance_absolute_gap(
                data,
                output_dir=output_dir,
                mode=mode,
                formats=formats,
                dpi=dpi,
            )
        )
        if include_detail:
            for outcome in (*MARKET_RELATIVE_GAP_OUTCOMES, *MECHANISM_RELATIVE_GAP_OUTCOMES):
                paths.extend(
                    plot_relative_gap_detail(
                        data,
                        outcome=outcome,
                        output_dir=output_dir,
                        mode=mode,
                        formats=formats,
                        dpi=dpi,
                    )
                )
    return tuple(paths)
