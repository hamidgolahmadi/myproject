"""Legacy OAT plotting utilities for the gamma experiment."""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = "oat_summary/oat_summary_by_topology.csv"
GAP_FILE = "oat_topology_comparison/topology_gap_by_parameter_value.csv"
PARAM_NAME = "gamma"
BASE_OUTPUT = "oat_plots"
TEMP_OUTPUT = "oat_plots_gamma"


def plot_gamma_oat(
    input_file=INPUT_FILE,
    gap_file=GAP_FILE,
    base_output=BASE_OUTPUT,
    temp_output=TEMP_OUTPUT,
):
    """Generate the original gamma OAT plots and move them into place."""
    final_output = os.path.join(base_output, PARAM_NAME)

    os.makedirs(base_output, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)

    df = pd.read_csv(input_file)
    gap_df = pd.read_csv(gap_file)

    df = df[df["varied_parameter"] == PARAM_NAME]
    gap_df = gap_df[gap_df["varied_parameter"] == PARAM_NAME]

    topologies = df["topology"].unique()

    plt.figure(figsize=(10, 6))

    for topo in topologies:
        sub = (
            df[df["topology"] == topo]
            .sort_values("varied_value")
        )
        plt.plot(
            sub["varied_value"],
            sub["mean_abs_return"],
            label=topo,
        )

    plt.xscale("log")
    plt.xlabel("gamma (log scale)")
    plt.ylabel("Mean Absolute Return")
    plt.title("Effect of gamma across topologies")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{temp_output}/gamma_mean_abs_return.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    gap_df_sorted = gap_df.sort_values("varied_value")

    plt.plot(
        gap_df_sorted["varied_value"],
        gap_df_sorted["mean_abs_return_range"],
    )

    plt.xscale("log")
    plt.xlabel("gamma (log scale)")
    plt.ylabel("Topology Gap (mean_abs_return)")
    plt.title("Topology Gap vs gamma")
    plt.grid(True)

    plt.savefig(
        f"{temp_output}/gamma_gap.png",
        dpi=300,
    )
    plt.close()

    pivot = df.pivot_table(
        index="topology",
        columns="varied_value",
        values="mean_abs_return",
    )

    pivot = pivot.reindex(
        sorted(pivot.columns),
        axis=1,
    )

    plt.figure(figsize=(14, 4))
    plt.imshow(
        pivot,
        aspect="auto",
    )
    plt.colorbar(label="Mean Absolute Return")

    plt.yticks(
        range(len(pivot.index)),
        pivot.index,
    )

    x_ticks = np.linspace(
        0,
        len(pivot.columns) - 1,
        6,
    ).astype(int)

    plt.xticks(
        x_ticks,
        [
            f"{pivot.columns[i]:.2e}"
            for i in x_ticks
        ],
        rotation=45,
    )

    plt.title("Heatmap of gamma effect across topologies")
    plt.tight_layout()

    plt.savefig(
        f"{temp_output}/gamma_heatmap.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 6))

    for topo in topologies:
        sub = (
            df[df["topology"] == topo]
            .sort_values("varied_value")
        )

        plt.plot(
            sub["varied_value"],
            sub["mean_belief_var"],
            label=topo,
        )

    plt.xscale("log")
    plt.xlabel("gamma (log scale)")
    plt.ylabel("Mean Belief Variance")
    plt.title("Belief Variance vs gamma")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{temp_output}/gamma_belief_variance.png",
        dpi=300,
    )
    plt.close()

    if os.path.exists(final_output):
        shutil.rmtree(final_output)

    shutil.move(
        temp_output,
        final_output,
    )

    print(
        f"All gamma plots saved in: {final_output}"
    )
