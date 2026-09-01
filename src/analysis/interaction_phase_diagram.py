"""Build legacy topology-relevance phase diagrams for ABG experiments."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RETURN_THRESHOLD = 1.10
BELIEF_THRESHOLD = 0.02

ALPHA_MIN = 0.125
ALPHA_MAX = 0.250
EPS = 1e-12


def build_interaction_phase_diagram(
    input_csv,
    outdir,
    return_threshold=RETURN_THRESHOLD,
    belief_threshold=BELIEF_THRESHOLD,
    alpha_min=ALPHA_MIN,
    alpha_max=ALPHA_MAX,
):
    """Build the legacy alpha phase-diagram tables and plots."""
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_cols = [
        "alpha_social",
        "beta",
        "gamma",
        "topology",
        "mean_abs_return",
        "mean_belief_var",
    ]

    missing = [
        column
        for column in required_cols
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        (df["alpha_social"] >= alpha_min)
        & (df["alpha_social"] <= alpha_max)
    ].copy()

    # Build topology gaps for each parameter cell.
    group_cols = ["alpha_social", "beta", "gamma"]
    gap_rows = []

    for keys, group in df.groupby(group_cols):
        alpha_val, beta_val, gamma_val = keys

        ret_vals = group[
            "mean_abs_return"
        ].to_numpy(dtype=float)

        belief_vals = group[
            "mean_belief_var"
        ].to_numpy(dtype=float)

        ret_max = np.max(ret_vals)
        ret_min = np.min(ret_vals)
        ret_mean = np.mean(ret_vals)

        belief_max = np.max(belief_vals)
        belief_min = np.min(belief_vals)
        belief_mean = np.mean(belief_vals)

        ret_range = ret_max - ret_min
        belief_range = belief_max - belief_min

        ret_rel_gap = ret_range / (ret_mean + EPS)
        belief_rel_gap = (
            belief_range / (belief_mean + EPS)
        )

        gap_rows.append(
            {
                "alpha_social": alpha_val,
                "beta": beta_val,
                "gamma": gamma_val,
                "mean_abs_return_range": ret_range,
                "mean_abs_return_relative_gap": (
                    ret_rel_gap
                ),
                "mean_belief_var_range": belief_range,
                "mean_belief_var_relative_gap": (
                    belief_rel_gap
                ),
            }
        )

    gap_df = (
        pd.DataFrame(gap_rows)
        .sort_values(
            ["alpha_social", "beta", "gamma"]
        )
    )

    gap_csv = os.path.join(
        outdir,
        "interaction_abg_gap_summary.csv",
    )
    gap_df.to_csv(gap_csv, index=False)
    print(f"Saved: {gap_csv}")

    # Aggregate topology-relevance indicators by alpha.
    phase_rows = []

    for alpha_val, group in gap_df.groupby(
        "alpha_social"
    ):
        total_cells = len(group)

        return_cells = (
            group["mean_abs_return_relative_gap"]
            > return_threshold
        ).sum()

        belief_cells = (
            group["mean_belief_var_relative_gap"]
            > belief_threshold
        ).sum()

        phase_rows.append(
            {
                "alpha_social": alpha_val,
                "total_cells": total_cells,
                "return_cells_above_threshold": int(
                    return_cells
                ),
                "belief_cells_above_threshold": int(
                    belief_cells
                ),
                "return_share": (
                    return_cells / total_cells
                    if total_cells > 0
                    else 0.0
                ),
                "belief_share": (
                    belief_cells / total_cells
                    if total_cells > 0
                    else 0.0
                ),
                "mean_return_relative_gap": group[
                    "mean_abs_return_relative_gap"
                ].mean(),
                "mean_belief_relative_gap": group[
                    "mean_belief_var_relative_gap"
                ].mean(),
            }
        )

    phase_df = (
        pd.DataFrame(phase_rows)
        .sort_values("alpha_social")
    )

    phase_csv = os.path.join(
        outdir,
        "alpha_phase_diagram_summary.csv",
    )
    phase_df.to_csv(phase_csv, index=False)
    print(f"Saved: {phase_csv}")

    # Plot shares above the legacy thresholds.
    plt.figure(figsize=(9, 5))
    plt.plot(
        phase_df["alpha_social"],
        phase_df["return_share"],
        marker="o",
        label="Return gap share",
    )
    plt.plot(
        phase_df["alpha_social"],
        phase_df["belief_share"],
        marker="o",
        label="Belief gap share",
    )
    plt.xlabel("alpha_social")
    plt.ylabel(
        "Share of (beta, gamma) cells above threshold"
    )
    plt.title(
        "Topology-relevance phase diagram across alpha"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot1 = os.path.join(
        outdir,
        "alpha_phase_diagram_shares.png",
    )
    plt.savefig(plot1, dpi=200)
    plt.close()
    print(f"Saved: {plot1}")

    # Plot average relative topology gaps.
    plt.figure(figsize=(9, 5))
    plt.plot(
        phase_df["alpha_social"],
        phase_df["mean_return_relative_gap"],
        marker="o",
        label="Mean return relative gap",
    )
    plt.plot(
        phase_df["alpha_social"],
        phase_df["mean_belief_relative_gap"],
        marker="o",
        label="Mean belief relative gap",
    )
    plt.xlabel("alpha_social")
    plt.ylabel(
        "Average relative gap over (beta, gamma)"
    )
    plt.title("Average topology gap across alpha")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot2 = os.path.join(
        outdir,
        "alpha_phase_diagram_mean_gaps.png",
    )
    plt.savefig(plot2, dpi=200)
    plt.close()
    print(f"Saved: {plot2}")

    return gap_df, phase_df
