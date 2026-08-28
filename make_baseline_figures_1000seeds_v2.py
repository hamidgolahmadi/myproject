"""
make_baseline_figures_1000seeds_v2.py

Creates paper figures for the 1000-seed baseline experiment.

Figures produced:

1. Distribution of systemic risk (riskS_mean)
2. Distribution of return volatility
3. Explosion probability by topology
4. Fraction of time stable comparison

All figures saved to:

baseline_no_policy_1000seeds_v2/figures/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = "baseline_no_policy_1000seeds_v2/merged/baseline_summary_by_seed_1000seeds_v2.csv"
OUTDIR = "baseline_no_policy_1000seeds_v2/figures"

os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)

df["explosive"] = df["riskS_mean"] > 1000

sns.set(style="whitegrid")

# =========================================================
# 1 Risk distribution
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="riskS_mean",
    hue="topology",
    bins=60,
    element="step",
    stat="density",
    common_norm=False
)

plt.xlim(0,200)
plt.title("Distribution of Systemic Risk")
plt.xlabel("Mean Systemic Risk (riskS)")
plt.ylabel("Density")

plt.savefig(f"{OUTDIR}/risk_distribution.png", dpi=300)
plt.close()

# =========================================================
# 2 Volatility distribution
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="return_vol",
    hue="topology",
    bins=60,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Distribution of Return Volatility")
plt.xlabel("Return Volatility")
plt.ylabel("Density")

plt.savefig(f"{OUTDIR}/volatility_distribution.png", dpi=300)
plt.close()

# =========================================================
# 3 Explosion probability
# =========================================================

explosion_rate = (
    df.groupby("topology")["explosive"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(6,4))

sns.barplot(
    data=explosion_rate,
    x="topology",
    y="explosive"
)

plt.ylabel("Explosion Probability")
plt.xlabel("Topology")

plt.savefig(f"{OUTDIR}/explosion_probability.png", dpi=300)
plt.close()

# =========================================================
# 4 Stability comparison
# =========================================================

plt.figure(figsize=(6,4))

sns.boxplot(
    data=df,
    x="topology",
    y="fraction_time_stable"
)

plt.ylabel("Fraction of Time Stable")

plt.savefig(f"{OUTDIR}/stability_comparison.png", dpi=300)
plt.close()

print("Figures saved to:", OUTDIR)
