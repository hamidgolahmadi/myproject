"""
make_paper_table_baseline_1000seeds_v2.py

Purpose
-------
This script builds the main paper-ready summary table for the large-scale
baseline experiment with 1000 seeds per topology.

Input
-----
It reads the merged seed-level results produced by:

    merge_baseline_no_policy_1000seeds_v2.py

Specifically, it uses:

    baseline_no_policy_1000seeds_v2/merged/baseline_summary_by_seed_1000seeds_v2.csv

This file contains one row per (topology, seed) simulation run.

What this script does
---------------------
1. Loads the merged seed-level baseline results.
2. Identifies "explosive" runs using a systemic-risk threshold.
3. Computes robust topology-level statistics that are more suitable for paper
   presentation than simple means.
4. Produces:
   - a CSV summary table,
   - a LaTeX table for direct use in Overleaf,
   - a console summary for quick inspection.

Why this is needed
------------------
In the baseline model, some runs can become explosive and create extremely
large values for risk and volatility. These outliers can distort mean-based
summaries. For this reason, the script reports:

- number of runs,
- number of explosive runs,
- explosion rate,
- median systemic risk (non-explosive runs only),
- median return volatility (non-explosive runs only),
- median time to stability (non-explosive runs only),
- mean fraction of stable time (all runs),
- mean network gini (all runs).

This produces a more interpretable paper table and better reflects the
underlying distribution of outcomes.

Outputs
-------
The script saves results in:

    baseline_no_policy_1000seeds_v2/merged/

with the following files:

1. paper_table_baseline_1000seeds_v2.csv
   Main topology-level paper table in CSV format

2. paper_table_baseline_1000seeds_v2.tex
   LaTeX version of the same table for direct inclusion in the paper

3. paper_table_baseline_1000seeds_v2_full.csv
   Extended version with extra columns useful for robustness checks

Notes
-----
A run is classified as "explosive" if:

    riskS_mean > 1000

This threshold matches the earlier diagnostic rule used in the project.
You may later change this threshold if you want to test robustness.
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "baseline_no_policy_1000seeds_v2/merged/baseline_summary_by_seed_1000seeds_v2.csv"
OUT_DIR = "baseline_no_policy_1000seeds_v2/merged"

OUTPUT_CSV_MAIN = os.path.join(OUT_DIR, "paper_table_baseline_1000seeds_v2.csv")
OUTPUT_CSV_FULL = os.path.join(OUT_DIR, "paper_table_baseline_1000seeds_v2_full.csv")
OUTPUT_TEX = os.path.join(OUT_DIR, "paper_table_baseline_1000seeds_v2.tex")

# Rule used to classify explosive runs
EXPLOSION_THRESHOLD = 1000.0

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_cols = [
    "topology",
    "seed",
    "riskS_mean",
    "peak_riskS",
    "belief_var_mean",
    "peak_belief_var",
    "return_vol",
    "cum_abs_returns",
    "cum_flow2",
    "gini_mean",
    "fraction_time_stable",
    "time_to_stability"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Clean time_to_stability in case blank values exist
df["time_to_stability"] = pd.to_numeric(df["time_to_stability"], errors="coerce")

# ============================================================
# EXPLOSION FLAG
# ============================================================

df["explosive"] = df["riskS_mean"] > EXPLOSION_THRESHOLD
df["non_explosive"] = ~df["explosive"]

# ============================================================
# BUILD PAPER TABLE
# ============================================================

rows = []

for topology, g in df.groupby("topology", sort=True):
    g_nonexp = g[g["non_explosive"]].copy()

    n_runs = len(g)
    n_explosive = int(g["explosive"].sum())
    explosion_rate = n_explosive / n_runs if n_runs > 0 else np.nan

    row = {
        "topology": topology,
        "n_runs": n_runs,
        "n_explosive": n_explosive,
        "explosion_rate": explosion_rate,

        # all runs
        "mean_fraction_time_stable_all": g["fraction_time_stable"].mean(),
        "mean_gini_all": g["gini_mean"].mean(),

        # non-explosive runs only
        "median_riskS_mean_nonexpl": g_nonexp["riskS_mean"].median(),
        "median_peak_riskS_nonexpl": g_nonexp["peak_riskS"].median(),
        "median_return_vol_nonexpl": g_nonexp["return_vol"].median(),
        "median_cum_abs_returns_nonexpl": g_nonexp["cum_abs_returns"].median(),
        "median_cum_flow2_nonexpl": g_nonexp["cum_flow2"].median(),
        "median_belief_var_mean_nonexpl": g_nonexp["belief_var_mean"].median(),
        "median_time_to_stability_nonexpl": g_nonexp["time_to_stability"].median(),

        # optional extra summary
        "mean_riskS_mean_nonexpl": g_nonexp["riskS_mean"].mean(),
        "std_riskS_mean_nonexpl": g_nonexp["riskS_mean"].std(),
        "mean_return_vol_nonexpl": g_nonexp["return_vol"].mean(),
        "std_return_vol_nonexpl": g_nonexp["return_vol"].std(),
    }

    rows.append(row)

summary = pd.DataFrame(rows).sort_values("topology").reset_index(drop=True)

# ============================================================
# MAIN PAPER TABLE (short version)
# ============================================================

paper_table = summary[
    [
        "topology",
        "n_runs",
        "n_explosive",
        "explosion_rate",
        "median_riskS_mean_nonexpl",
        "median_return_vol_nonexpl",
        "median_time_to_stability_nonexpl",
        "mean_fraction_time_stable_all",
        "mean_gini_all",
    ]
].copy()

paper_table = paper_table.rename(
    columns={
        "topology": "Topology",
        "n_runs": "Runs",
        "n_explosive": "Explosive Runs",
        "explosion_rate": "Explosion Rate",
        "median_riskS_mean_nonexpl": "Median RiskS",
        "median_return_vol_nonexpl": "Median Return Vol",
        "median_time_to_stability_nonexpl": "Median Time to Stability",
        "mean_fraction_time_stable_all": "Mean Fraction Stable",
        "mean_gini_all": "Mean Gini",
    }
)

# nicer formatting values in CSV export
paper_table_rounded = paper_table.copy()
for col in [
    "Explosion Rate",
    "Median RiskS",
    "Median Return Vol",
    "Median Time to Stability",
    "Mean Fraction Stable",
    "Mean Gini",
]:
    paper_table_rounded[col] = paper_table_rounded[col].astype(float).round(4)

# save CSVs
paper_table_rounded.to_csv(OUTPUT_CSV_MAIN, index=False)
summary.to_csv(OUTPUT_CSV_FULL, index=False)

# ============================================================
# BUILD LATEX TABLE
# ============================================================

latex_table = paper_table_rounded.to_latex(
    index=False,
    caption="Baseline stability metrics across network topologies (1000 seeds per topology). Median metrics are computed using non-explosive runs only.",
    label="tab:baseline_1000seeds",
    float_format="%.4f",
    escape=False
)

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write(latex_table)

# ============================================================
# PRINT CONSOLE OUTPUT
# ============================================================

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("\n=== PAPER TABLE: BASELINE (1000 SEEDS) ===\n")
print(paper_table_rounded.to_string(index=False))

print("\nSaved files:")
print(f"  - {OUTPUT_CSV_MAIN}")
print(f"  - {OUTPUT_CSV_FULL}")
print(f"  - {OUTPUT_TEX}")

print("\nQuick validation:")
print(df.groupby("topology").size())
print(f"\nTotal rows in source data: {len(df)}")
print(f"Explosion threshold used: riskS_mean > {EXPLOSION_THRESHOLD}")
