"""
merge_baseline_no_policy_1000seeds_v2.py

Purpose
-------
This script merges the results of the large-scale baseline simulations executed
with SLURM array jobs. The baseline experiment was run with 1000 random seeds,
but due to HPC scheduling constraints the simulations were split into 100
independent chunks (each running 10 seeds).

Each chunk produced its own set of CSV output files inside:

    baseline_no_policy_1000seeds_v2/chunk_<id>/

where <id> = 0 ... 99.

Each chunk directory contains:

    baseline_raw_v2.csv
    baseline_summary_by_seed_v2.csv
    baseline_summary_v2.csv
    stability_threshold_v2.csv

This script performs the following tasks:

1. Collects all chunk-level CSV files across the 100 SLURM jobs.
2. Concatenates the seed-level summaries into a single dataset.
3. Optionally merges the raw simulation panel data.
4. Recomputes the topology-level summary statistics from the merged seed-level data.
5. Stores the final merged outputs inside:

       baseline_no_policy_1000seeds_v2/merged/

Outputs
-------

baseline_summary_by_seed_1000seeds_v2.csv
    Seed-level results across all seeds and topologies
    (~3000 rows = 1000 seeds × 3 network topologies)

baseline_raw_1000seeds_v2.csv
    Optional merged raw simulation panel (can be large)

baseline_summary_1000seeds_v2.csv
    Topology-level averages recomputed from the merged seed-level results

stability_threshold_v2.csv
    Copy of the stability criteria used in the experiment

Notes
-----

The baseline simulations evaluate systemic risk dynamics in different
network topologies (random, scale-free, small-world). Because SLURM array
jobs were used for scalability, the outputs must be merged before statistical
analysis or paper tables can be generated.

This script is intended to be run once after all SLURM jobs have completed.
"""

import os
import glob
import pandas as pd

BASE_DIR = "baseline_no_policy_1000seeds_v2"
OUT_DIR = os.path.join(BASE_DIR, "merged")
os.makedirs(OUT_DIR, exist_ok=True)

summary_by_seed_files = sorted(
    glob.glob(os.path.join(BASE_DIR, "chunk_*", "baseline_summary_by_seed_v2.csv"))
)

summary_files = sorted(
    glob.glob(os.path.join(BASE_DIR, "chunk_*", "baseline_summary_v2.csv"))
)

raw_files = sorted(
    glob.glob(os.path.join(BASE_DIR, "chunk_*", "baseline_raw_v2.csv"))
)

threshold_files = sorted(
    glob.glob(os.path.join(BASE_DIR, "chunk_*", "stability_threshold_v2.csv"))
)

print(f"Found {len(summary_by_seed_files)} summary_by_seed files")
print(f"Found {len(summary_files)} summary files")
print(f"Found {len(raw_files)} raw files")
print(f"Found {len(threshold_files)} threshold files")

if len(summary_by_seed_files) == 0:
    raise FileNotFoundError("No baseline_summary_by_seed_v2.csv files found.")

# -------------------------
# Merge summary_by_seed
# -------------------------
dfs_seed = [pd.read_csv(f) for f in summary_by_seed_files]
merged_seed = pd.concat(dfs_seed, ignore_index=True)

merged_seed.to_csv(
    os.path.join(OUT_DIR, "baseline_summary_by_seed_1000seeds_v2.csv"),
    index=False
)

print("Saved merged seed-level summary.")

# -------------------------
# Merge raw
# -------------------------
if len(raw_files) > 0:
    dfs_raw = [pd.read_csv(f) for f in raw_files]
    merged_raw = pd.concat(dfs_raw, ignore_index=True)
    merged_raw.to_csv(
        os.path.join(OUT_DIR, "baseline_raw_1000seeds_v2.csv"),
        index=False
    )
    print("Saved merged raw panel.")
else:
    print("No raw files found.")

# -------------------------
# Recompute topology summary from merged seed-level data
# -------------------------
# Columns expected in baseline_summary_by_seed_v2.csv:
# topology, seed, riskS_mean, peak_riskS, belief_var_mean, peak_belief_var,
# return_vol, cum_abs_returns, cum_flow2, gini_mean, fraction_time_stable, time_to_stability

required_cols = [
    "topology", "seed", "riskS_mean", "peak_riskS", "belief_var_mean",
    "peak_belief_var", "return_vol", "cum_abs_returns", "cum_flow2",
    "gini_mean", "fraction_time_stable", "time_to_stability"
]

missing = [c for c in required_cols if c not in merged_seed.columns]
if missing:
    raise ValueError(f"Missing expected columns in merged seed file: {missing}")

topology_summary = (
    merged_seed
    .groupby("topology", as_index=False)
    .agg({
        "riskS_mean": "mean",
        "peak_riskS": "mean",
        "belief_var_mean": "mean",
        "peak_belief_var": "mean",
        "return_vol": "mean",
        "cum_abs_returns": "mean",
        "cum_flow2": "mean",
        "gini_mean": "mean",
        "fraction_time_stable": "mean",
        "time_to_stability": "mean"
    })
)

topology_summary.to_csv(
    os.path.join(OUT_DIR, "baseline_summary_1000seeds_v2.csv"),
    index=False
)

print("Saved merged topology-level summary.")

# -------------------------
# Copy one threshold file
# -------------------------
if len(threshold_files) > 0:
    thresh = pd.read_csv(threshold_files[0])
    thresh.to_csv(
        os.path.join(OUT_DIR, "stability_threshold_v2.csv"),
        index=False
    )
    print("Saved threshold file.")
else:
    print("No threshold file found.")

# -------------------------
# Quick checks
# -------------------------
print("\nRow counts by topology in merged seed summary:")
print(merged_seed.groupby("topology").size())

print("\nExpected rows if complete: 1000 per topology")
print("\nTotal rows in merged seed summary:", len(merged_seed))
