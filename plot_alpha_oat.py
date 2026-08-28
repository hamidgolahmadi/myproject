import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load summary
# =========================
df = pd.read_csv("oat_summary/oat_summary_by_topology.csv")

df = df[df["varied_parameter"] == "alpha_social"].copy()

# =========================
# Plot 1: mean_abs_return vs alpha
# =========================
plt.figure(figsize=(10,6))

for topo in ["random", "small_world", "scale_free"]:
    sub = df[df["topology"] == topo]
    plt.plot(sub["varied_value"], sub["mean_abs_return"], label=topo)

plt.xscale("log")
plt.xlabel("alpha_social (log scale)")
plt.ylabel("Mean Absolute Return")
plt.title("Effect of alpha_social on volatility")
plt.legend()
plt.grid(True)

plt.savefig("oat_plots/alpha/alpha_mean_abs_return.png", dpi=150)
plt.close()

# =========================
# Plot 2: belief variance
# =========================
plt.figure(figsize=(10,6))

for topo in ["random", "small_world", "scale_free"]:
    sub = df[df["topology"] == topo]
    plt.plot(sub["varied_value"], sub["mean_belief_var"], label=topo)

plt.xscale("log")
plt.xlabel("alpha_social (log scale)")
plt.ylabel("Mean Belief Variance")
plt.title("Effect of alpha_social on belief dispersion")
plt.legend()
plt.grid(True)

plt.savefig("oat_plots/alpha/alpha_belief_var.png", dpi=150)
plt.close()

# =========================
# Plot 3: GAP (topology difference)
# =========================
gap_df = pd.read_csv("oat_topology_comparison/topology_gap_by_parameter_value.csv")
gap_df = gap_df[gap_df["varied_parameter"] == "alpha_social"]

plt.figure(figsize=(10,6))

plt.plot(gap_df["varied_value"], gap_df["mean_abs_return_range"], label="return gap")
plt.plot(gap_df["varied_value"], gap_df["mean_belief_var_range"], label="belief gap")

plt.xscale("log")
plt.xlabel("alpha_social (log scale)")
plt.ylabel("Gap across topologies")
plt.title("Where network structure starts to matter")
plt.legend()
plt.grid(True)

plt.savefig("oat_plots/alpha/alpha_gap.png", dpi=150)
plt.close()

# =========================
# Plot 4: Heatmap (important)
# =========================
import seaborn as sns

pivot = df.pivot_table(
    index="topology",
    columns="varied_value",
    values="mean_abs_return"
)

plt.figure(figsize=(14,4))
sns.heatmap(pivot, cmap="viridis")

plt.title("Heatmap: volatility vs alpha_social")
plt.xlabel("alpha_social")
plt.ylabel("Topology")

plt.savefig("oat_plots/alpha/alpha_heatmap.png", dpi=150)
plt.close()

print("Plots saved successfully.")
