"""
make_journal_figures_baseline_1000seeds_v2.py

Purpose
-------
Create journal-quality figures for the baseline experiment with 1000 seeds per topology.

This script improves on earlier exploratory plots by:
1. removing explosive runs from the risk-distribution figure,
2. using a log scale for volatility,
3. adding confidence intervals to explosion probabilities,
4. using violin plots for stability comparisons.

Input
-----
baseline_no_policy_1000seeds_v2/merged/baseline_summary_by_seed_1000seeds_v2.csv

Output directory
----------------
baseline_no_policy_1000seeds_v2/figures_journal/

Figures created
---------------
1. risk_distribution_nonexplosive.png
2. volatility_distribution_logscale.png
3. explosion_probability_ci.png
4. stability_violin.png

Notes
-----
A run is classified as explosive if:
    riskS_mean > 1000

This threshold matches the project rule used in the baseline paper table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "baseline_no_policy_1000seeds_v2/merged/baseline_summary_by_seed_1000seeds_v2.csv"
OUTDIR = "baseline_no_policy_1000seeds_v2/figures_journal"
EXPLOSION_THRESHOLD = 1000.0

os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)
df["time_to_stability"] = pd.to_numeric(df["time_to_stability"], errors="coerce")
df["explosive"] = df["riskS_mean"] > EXPLOSION_THRESHOLD

# Clean labels for paper
label_map = {
    "random_fixed_extreme": "Random",
    "scale_free_extreme": "Scale-Free",
    "small_world_clustered": "Small-World"
}
df["Topology"] = df["topology"].map(label_map).fillna(df["topology"])

# Non-explosive subset
df_nonexp = df[~df["explosive"]].copy()

# ============================================================
# STYLE
# ============================================================

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

topology_order = ["Random", "Scale-Free", "Small-World"]

# ============================================================
# 1) CLEAN RISK DISTRIBUTION (NON-EXPLOSIVE ONLY)
# ============================================================

plt.figure(figsize=(8, 5.5))

for topo in topology_order:
    sub = df_nonexp[df_nonexp["Topology"] == topo]
    sns.kdeplot(
        data=sub,
        x="riskS_mean",
        fill=False,
        common_norm=False,
        linewidth=2,
        label=topo
    )

plt.title("Systemic Risk Distribution (Non-Explosive Runs Only)")
plt.xlabel("Mean Systemic Risk")
plt.ylabel("Density")
plt.legend(title="")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "risk_distribution_nonexplosive.png"))
plt.close()

# ============================================================
# 2) LOG-SCALE VOLATILITY DISTRIBUTION
# ============================================================

# Use all runs with positive volatility
df_vol = df[df["return_vol"] > 0].copy()

plt.figure(figsize=(8, 5.5))

for topo in topology_order:
    sub = df_vol[df_vol["Topology"] == topo]
    sns.kdeplot(
        data=sub,
        x="return_vol",
        fill=False,
        common_norm=False,
        linewidth=2,
        label=topo
    )

plt.xscale("log")
plt.title("Return Volatility Distribution (Log Scale)")
plt.xlabel("Return Volatility (log scale)")
plt.ylabel("Density")
plt.legend(title="")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "volatility_distribution_logscale.png"))
plt.close()

# ============================================================
# 3) EXPLOSION PROBABILITY WITH 95% CONFIDENCE INTERVALS
# ============================================================

grp = (
    df.groupby("Topology")["explosive"]
    .agg(ExplosiveRuns="sum", Runs="count")
    .reset_index()
)
grp["ExplosionProb"] = grp["ExplosiveRuns"] / grp["Runs"]

# Normal approximation CI for a binomial proportion
z = 1.96
grp["se"] = np.sqrt(grp["ExplosionProb"] * (1 - grp["ExplosionProb"]) / grp["Runs"])
grp["ci_low"] = (grp["ExplosionProb"] - z * grp["se"]).clip(lower=0)
grp["ci_high"] = (grp["ExplosionProb"] + z * grp["se"]).clip(upper=1)
grp["yerr_lower"] = grp["ExplosionProb"] - grp["ci_low"]
grp["yerr_upper"] = grp["ci_high"] - grp["ExplosionProb"]

grp = grp.set_index("Topology").loc[topology_order].reset_index()

plt.figure(figsize=(7, 5.5))
bars = plt.bar(
    grp["Topology"],
    grp["ExplosionProb"],
    yerr=np.vstack([grp["yerr_lower"], grp["yerr_upper"]]),
    capsize=6
)

plt.ylabel("Explosion Probability")
plt.xlabel("")
plt.title("Explosion Probability by Topology (95% CI)")
plt.ylim(0, min(1.0, grp["ci_high"].max() + 0.05))

# Add labels above bars
for i, row in grp.iterrows():
    plt.text(
        i,
        row["ExplosionProb"] + 0.01,
        f"{row['ExplosionProb']:.3f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "explosion_probability_ci.png"))
plt.close()

# ============================================================
# 4) VIOLIN PLOT FOR STABILITY
# ============================================================

plt.figure(figsize=(8, 5.5))

sns.violinplot(
    data=df,
    x="Topology",
    y="fraction_time_stable",
    order=topology_order,
    inner="box",
    cut=0
)

plt.title("Distribution of Stable Time Fraction")
plt.xlabel("")
plt.ylabel("Fraction of Time Stable")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "stability_violin.png"))
plt.close()

# ============================================================
# SAVE SUMMARY TABLE USED FOR FIGURES
# ============================================================

grp.to_csv(os.path.join(OUTDIR, "explosion_probability_ci_table.csv"), index=False)

print("Saved journal-quality figures to:", OUTDIR)
print("\nFiles created:")
print(" - risk_distribution_nonexplosive.png")
print(" - volatility_distribution_logscale.png")
print(" - explosion_probability_ci.png")
print(" - stability_violin.png")
print(" - explosion_probability_ci_table.csv")
