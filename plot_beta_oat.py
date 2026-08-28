import os
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_FILE = "oat_summary/oat_summary_by_topology.csv"
GAP_FILE = "oat_topology_comparison/topology_gap_by_parameter_value.csv"
OUTDIR = "beta_plots"


METRICS_TO_PLOT = [
    "explosion_rate",
    "mean_abs_return",
    "mean_belief_var",
    "mean_avg_abs_position",
]

GAP_METRICS = [
    "explosion_rate_range",
    "mean_abs_return_range",
    "mean_belief_var_range",
    "mean_avg_abs_position_range",
]


def save_beta_table():
    df = pd.read_csv(SUMMARY_FILE)
    df = df[df["varied_parameter"] == "beta"].copy()

    df = df.sort_values(["topology", "varied_value"])
    out_path = os.path.join(OUTDIR, "beta_summary_table.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved table: {out_path}")


def plot_metric_by_topology(metric):
    df = pd.read_csv(SUMMARY_FILE)
    df = df[df["varied_parameter"] == "beta"].copy()

    plt.figure(figsize=(8, 5))

    for topo in sorted(df["topology"].unique()):
        sub = df[df["topology"] == topo].copy().sort_values("varied_value")
        plt.plot(sub["varied_value"], sub[metric], marker="o", linewidth=1, markersize=3, label=topo)

    plt.xscale("log")
    plt.xlabel("beta")
    plt.ylabel(metric)
    plt.title(f"{metric} vs beta by topology")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTDIR, f"{metric}_vs_beta_by_topology.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_gap_metric(metric):
    df = pd.read_csv(GAP_FILE)
    df = df[df["varied_parameter"] == "beta"].copy().sort_values("varied_value")

    plt.figure(figsize=(8, 5))
    plt.plot(df["varied_value"], df[metric], marker="o", linewidth=1, markersize=3)

    plt.xscale("log")
    plt.xlabel("beta")
    plt.ylabel(metric)
    plt.title(f"{metric} vs beta")
    plt.tight_layout()

    out_path = os.path.join(OUTDIR, f"{metric}_vs_beta.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved gap plot: {out_path}")


def plot_heatmap(metric):
    df = pd.read_csv(SUMMARY_FILE)
    df = df[df["varied_parameter"] == "beta"].copy()

    pivot = df.pivot(index="topology", columns="varied_value", values=metric)
    pivot = pivot.sort_index(axis=0)
    pivot = pivot.sort_index(axis=1)

    plt.figure(figsize=(12, 3.5))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(pivot.index)), pivot.index)
    
    # برای beta چون 1000 sample داریم، همه tickها را نشان نده
    x_positions = list(range(0, len(pivot.columns), max(1, len(pivot.columns)//10)))
    x_labels = [f"{pivot.columns[i]:.3g}" for i in x_positions]
    plt.xticks(x_positions, x_labels, rotation=45)

    plt.xlabel("beta")
    plt.ylabel("topology")
    plt.title(f"Heatmap of {metric} across topology and beta")
    plt.colorbar()
    plt.tight_layout()

    out_path = os.path.join(OUTDIR, f"heatmap_{metric}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved heatmap: {out_path}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    save_beta_table()

    for metric in METRICS_TO_PLOT:
        plot_metric_by_topology(metric)

    for metric in GAP_METRICS:
        plot_gap_metric(metric)

    # heatmaps برای چند metric اصلی
    for metric in ["explosion_rate", "mean_abs_return", "mean_belief_var"]:
        plot_heatmap(metric)


if __name__ == "__main__":
    main()
