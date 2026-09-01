"""Legacy OAT plotting utilities for the alpha-social experiment."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_alpha_oat(
    summary_file="oat_summary/oat_summary_by_topology.csv",
    gap_file=(
        "oat_topology_comparison/"
        "topology_gap_by_parameter_value.csv"
    ),
    outdir="oat_plots/alpha",
):
    """Generate the original alpha-social OAT plots."""
    df = pd.read_csv(summary_file)
    df = df[df["varied_parameter"] == "alpha_social"].copy()

    plt.figure(figsize=(10, 6))

    for topo in ["random", "small_world", "scale_free"]:
        sub = df[df["topology"] == topo]
        plt.plot(
            sub["varied_value"],
            sub["mean_abs_return"],
            label=topo,
        )

    plt.xscale("log")
    plt.xlabel("alpha_social (log scale)")
    plt.ylabel("Mean Absolute Return")
    plt.title("Effect of alpha_social on volatility")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{outdir}/alpha_mean_abs_return.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for topo in ["random", "small_world", "scale_free"]:
        sub = df[df["topology"] == topo]
        plt.plot(
            sub["varied_value"],
            sub["mean_belief_var"],
            label=topo,
        )

    plt.xscale("log")
    plt.xlabel("alpha_social (log scale)")
    plt.ylabel("Mean Belief Variance")
    plt.title("Effect of alpha_social on belief dispersion")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{outdir}/alpha_belief_var.png",
        dpi=150,
    )
    plt.close()

    gap_df = pd.read_csv(gap_file)
    gap_df = gap_df[
        gap_df["varied_parameter"] == "alpha_social"
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        gap_df["varied_value"],
        gap_df["mean_abs_return_range"],
        label="return gap",
    )
    plt.plot(
        gap_df["varied_value"],
        gap_df["mean_belief_var_range"],
        label="belief gap",
    )

    plt.xscale("log")
    plt.xlabel("alpha_social (log scale)")
    plt.ylabel("Gap across topologies")
    plt.title("Where network structure starts to matter")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{outdir}/alpha_gap.png",
        dpi=150,
    )
    plt.close()

    pivot = df.pivot_table(
        index="topology",
        columns="varied_value",
        values="mean_abs_return",
    )

    plt.figure(figsize=(14, 4))
    sns.heatmap(pivot, cmap="viridis")

    plt.title("Heatmap: volatility vs alpha_social")
    plt.xlabel("alpha_social")
    plt.ylabel("Topology")

    plt.savefig(
        f"{outdir}/alpha_heatmap.png",
        dpi=150,
    )
    plt.close()

    print("Plots saved successfully.")
