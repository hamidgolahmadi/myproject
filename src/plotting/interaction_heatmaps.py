"""Heatmap plotting for legacy ABG interaction experiments."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOPOLOGIES = ["random", "small_world", "scale_free"]


def format_alpha(alpha):
    """Format alpha for legacy output filenames."""
    return str(alpha).replace(".", "p")


def plot_topology_heatmap(
    df,
    alpha_value,
    topology,
    metric,
    outdir,
):
    """Plot one topology-specific beta-gamma heatmap."""
    sub = df[
        (df["alpha_social"] == alpha_value)
        & (df["topology"] == topology)
    ].copy()

    pivot = sub.pivot(
        index="gamma",
        columns="beta",
        values=metric,
    )
    pivot = pivot.sort_index(axis=0)
    pivot = pivot.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    x_positions = np.arange(len(pivot.columns))
    y_positions = np.arange(len(pivot.index))

    x_show = np.linspace(
        0,
        len(x_positions) - 1,
        min(8, len(x_positions)),
        dtype=int,
    )
    y_show = np.linspace(
        0,
        len(y_positions) - 1,
        min(8, len(y_positions)),
        dtype=int,
    )

    ax.set_xticks(x_show)
    ax.set_xticklabels(
        [f"{pivot.columns[i]:.2g}" for i in x_show],
        rotation=45,
    )
    ax.set_yticks(y_show)
    ax.set_yticklabels(
        [f"{pivot.index[i]:.2g}" for i in y_show]
    )

    ax.set_xlabel("beta")
    ax.set_ylabel("gamma")
    ax.set_title(
        f"{metric} | alpha={alpha_value} | {topology}"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)

    plt.tight_layout()

    outpath = os.path.join(
        outdir,
        f"{metric}_alpha_{format_alpha(alpha_value)}_{topology}.png",
    )
    plt.savefig(outpath, dpi=180)
    plt.close()


def build_gap_df(df, alpha_value, metric):
    """Build legacy topology-gap statistics for one alpha."""
    sub = df[
        df["alpha_social"] == alpha_value
    ].copy()

    grouped = sub.groupby(["beta", "gamma"])

    rows = []

    for (beta, gamma), group in grouped:
        vals = group[metric].values

        row = {
            "alpha_social": alpha_value,
            "beta": beta,
            "gamma": gamma,
            f"{metric}_range": float(
                np.max(vals) - np.min(vals)
            ),
            f"{metric}_relative_gap": float(
                (np.max(vals) - np.min(vals))
                / (np.mean(vals) + 1e-12)
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_gap_heatmap(
    gap_df,
    alpha_value,
    metric_name,
    gap_col,
    outdir,
):
    """Plot one legacy beta-gamma topology-gap heatmap."""
    pivot = gap_df.pivot(
        index="gamma",
        columns="beta",
        values=gap_col,
    )
    pivot = pivot.sort_index(axis=0)
    pivot = pivot.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
    )

    x_positions = np.arange(len(pivot.columns))
    y_positions = np.arange(len(pivot.index))

    x_show = np.linspace(
        0,
        len(x_positions) - 1,
        min(8, len(x_positions)),
        dtype=int,
    )
    y_show = np.linspace(
        0,
        len(y_positions) - 1,
        min(8, len(y_positions)),
        dtype=int,
    )

    ax.set_xticks(x_show)
    ax.set_xticklabels(
        [f"{pivot.columns[i]:.2g}" for i in x_show],
        rotation=45,
    )
    ax.set_yticks(y_show)
    ax.set_yticklabels(
        [f"{pivot.index[i]:.2g}" for i in y_show]
    )

    ax.set_xlabel("beta")
    ax.set_ylabel("gamma")
    ax.set_title(
        f"{gap_col} | alpha={alpha_value}"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(gap_col)

    plt.tight_layout()

    outpath = os.path.join(
        outdir,
        f"{gap_col}_alpha_{format_alpha(alpha_value)}.png",
    )
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_interaction_heatmaps(
    summary_file="interaction_summary/interaction_abg_summary.csv",
    outdir="interaction_plots",
):
    """Generate all legacy ABG interaction heatmaps."""
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(summary_file)

    alpha_values = sorted(
        df["alpha_social"].unique()
    )

    for alpha_value in alpha_values:
        for topology in TOPOLOGIES:
            plot_topology_heatmap(
                df=df,
                alpha_value=alpha_value,
                topology=topology,
                metric="mean_abs_return",
                outdir=outdir,
            )

    for alpha_value in alpha_values:
        gap_return = build_gap_df(
            df,
            alpha_value,
            "mean_abs_return",
        )

        plot_gap_heatmap(
            gap_df=gap_return,
            alpha_value=alpha_value,
            metric_name="mean_abs_return",
            gap_col="mean_abs_return_range",
            outdir=outdir,
        )
        plot_gap_heatmap(
            gap_df=gap_return,
            alpha_value=alpha_value,
            metric_name="mean_abs_return",
            gap_col="mean_abs_return_relative_gap",
            outdir=outdir,
        )

        gap_belief = build_gap_df(
            df,
            alpha_value,
            "mean_belief_var",
        )

        plot_gap_heatmap(
            gap_df=gap_belief,
            alpha_value=alpha_value,
            metric_name="mean_belief_var",
            gap_col="mean_belief_var_range",
            outdir=outdir,
        )
        plot_gap_heatmap(
            gap_df=gap_belief,
            alpha_value=alpha_value,
            metric_name="mean_belief_var",
            gap_col="mean_belief_var_relative_gap",
            outdir=outdir,
        )

    print(f"Saved plots to: {outdir}")
