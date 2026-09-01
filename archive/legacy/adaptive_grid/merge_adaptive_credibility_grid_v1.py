# -*- coding: utf-8 -*-
"""
merge_adaptive_credibility_grid_v1.py

Merge all Paper 1 chunk-level CSV files into:
- merged_seed_level.csv
- merged_grid_agg.csv
"""

import os
import glob
import pandas as pd

BASE_DIR = "adaptive_credibility_grid_v1"
OUT_DIR = os.path.join(BASE_DIR, "merged")
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(BASE_DIR, "grid_topo_*.csv")))
if not files:
    raise FileNotFoundError("No adaptive credibility grid csv files found.")

dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)

merged.to_csv(os.path.join(OUT_DIR, "merged_seed_level.csv"), index=False)

agg = (
    merged.groupby(["topology", "beta", "gamma"], as_index=False)
    .agg(
        return_vol_mean=("return_vol", "mean"),
        return_vol_median=("return_vol", "median"),
        riskS_mean=("riskS_mean", "mean"),
        riskS_median=("riskS_mean", "median"),
        belief_var_mean=("belief_var_mean", "mean"),
        fraction_time_stable_mean=("fraction_time_stable", "mean"),
        fraction_time_stable_median=("fraction_time_stable", "median"),
        explosion_probability=("explosive", "mean"),
        time_to_stability_median=("time_to_stability", "median"),
        n_runs=("seed", "count"),
    )
)

agg.to_csv(os.path.join(OUT_DIR, "merged_grid_agg.csv"), index=False)

print("Saved:")
print(os.path.join(OUT_DIR, "merged_seed_level.csv"))
print(os.path.join(OUT_DIR, "merged_grid_agg.csv"))
print()
print(agg.head(12).to_string(index=False))
