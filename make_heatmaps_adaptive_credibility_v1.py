# -*- coding: utf-8 -*-
"""
make_heatmaps_adaptive_credibility_v1.py

Create Paper 1 heatmaps from merged adaptive credibility results.

For each topology, build heatmaps for:
1. return_vol_mean
2. explosion_probability
3. fraction_time_stable_mean
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = "adaptive_credibility_grid_v1/merged/merged_grid_agg.csv"
OUTDIR = "adaptive_credibility_grid_v1/figures"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)

label_map = {
    "random_fixed_extreme": "Random",
    "scale_free_extreme": "Scale-Free",
    "small_world_clustered": "Small-World",
}
df["TopologyLabel"] = df["topology"].map(label_map).fillna(df["topology"])

sns.set_theme(style="white", context="talk")

metrics = [
    ("return_vol_mean", "Return Volatility"),
    ("explosion_probability", "Explosion Probability"),
    ("fraction_time_stable_mean", "Fraction Time Stable"),
]

for topo in df["topology"].unique():
    sub = df[df["topology"] == topo].copy()
    topo_label = sub["TopologyLabel"].iloc[0]

    for metric_col, metric_title in metrics:
        pivot = sub.pivot(index="gamma", columns="beta", values=metric_col)
        pivot = pivot.sort_index(ascending=False)

        plt.figure(figsize=(9, 6))
        sns.heatmap(pivot, cmap="viridis")
        plt.title(f"{topo_label}: {metric_title}")
        plt.xlabel("beta")
        plt.ylabel("gamma")
        plt.tight_layout()

        fname = f"{topo}_{metric_col}_heatmap.png"
        plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
        plt.close()

print("Saved heatmaps to:", OUTDIR)
