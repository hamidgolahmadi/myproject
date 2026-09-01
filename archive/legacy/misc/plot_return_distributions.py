import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# paths
# =========================
BASELINE_FILE = "interaction_merged/interaction_abg_all_merged.csv"
OUTDIR = "paper_figures"

# =========================
# choose one (beta, gamma, alpha) point for conditional distribution
# the script will snap to the nearest available grid value
# =========================
TARGET_ALPHA = 0.15
TARGET_BETA = 10.0
TARGET_GAMMA = 0.2

os.makedirs(OUTDIR, exist_ok=True)


def nearest_value(values, target):
    arr = np.array(sorted(pd.unique(values)), dtype=float)
    return float(arr[np.argmin(np.abs(arr - target))])


def main():
    df = pd.read_csv(BASELINE_FILE)

    required_cols = [
        "topology",
        "seed",
        "alpha_social",
        "beta",
        "gamma",
        "mean_abs_return",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ---------------------------------
    # Figure 1: baseline distribution across topologies
    # Here "baseline" means pooled over all parameter combinations
    # ---------------------------------
    topo_order = ["random", "small_world", "scale_free"]
    baseline_data = [
        df.loc[df["topology"] == topo, "mean_abs_return"].dropna().values
        for topo in topo_order
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(baseline_data, labels=topo_order, showfliers=False)
    plt.ylabel("Mean absolute return")
    plt.title("Baseline return distribution by topology")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "baseline_return_distribution_by_topology.png"),
        dpi=220
    )
    plt.close()

    # ---------------------------------
    # Figure 2: selected (alpha, beta, gamma) distribution across topologies
    # ---------------------------------
    alpha_star = nearest_value(df["alpha_social"], TARGET_ALPHA)
    beta_star = nearest_value(df["beta"], TARGET_BETA)
    gamma_star = nearest_value(df["gamma"], TARGET_GAMMA)

    sub = df[
        (df["alpha_social"] == alpha_star) &
        (df["beta"] == beta_star) &
        (df["gamma"] == gamma_star)
    ].copy()

    if sub.empty:
        raise ValueError("Selected alpha-beta-gamma combination returned no rows.")

    conditional_data = [
        sub.loc[sub["topology"] == topo, "mean_abs_return"].dropna().values
        for topo in topo_order
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(conditional_data, labels=topo_order, showfliers=False)
    plt.ylabel("Mean absolute return")
    plt.title(
        f"Return distribution by topology\n"
        f"alpha={alpha_star:.3g}, beta={beta_star:.3g}, gamma={gamma_star:.3g}"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "return_distribution_selected_beta_gamma.png"),
        dpi=220
    )
    plt.close()

    # ---------------------------------
    # small text summary
    # ---------------------------------
    summary = pd.DataFrame({
        "topology": topo_order,
        "baseline_n": [len(x) for x in baseline_data],
        "baseline_mean": [float(np.mean(x)) if len(x) else np.nan for x in baseline_data],
        "baseline_std": [float(np.std(x)) if len(x) else np.nan for x in baseline_data],
        "selected_n": [len(x) for x in conditional_data],
        "selected_mean": [float(np.mean(x)) if len(x) else np.nan for x in conditional_data],
        "selected_std": [float(np.std(x)) if len(x) else np.nan for x in conditional_data],
    })
    summary.to_csv(
        os.path.join(OUTDIR, "return_distribution_summary.csv"),
        index=False
    )

    print("Saved:")
    print(os.path.join(OUTDIR, "baseline_return_distribution_by_topology.png"))
    print(os.path.join(OUTDIR, "return_distribution_selected_beta_gamma.png"))
    print(os.path.join(OUTDIR, "return_distribution_summary.csv"))
    print(f"Selected point snapped to alpha={alpha_star}, beta={beta_star}, gamma={gamma_star}")


if __name__ == "__main__":
    main()
