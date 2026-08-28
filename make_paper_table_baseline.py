import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_CSV = "baseline_results_v2/baseline_summary_by_seed_v2.csv"
OUTPUT_CSV = "baseline_results_v2/paper_table_baseline_v2.csv"

# Explosive-run rule:
# Column 3 in your awk test was riskS_mean, and you used > 1000
EXPLOSION_THRESHOLD = 1000.0

# =========================
# LOAD
# =========================
df = pd.read_csv(INPUT_CSV)

# Expected columns from your file
required_cols = [
    "topology",
    "riskS_mean",
    "peak_riskS",
    "belief_var_mean",
    "peak_belief_var",
    "return_vol",
    "cum_abs_returns",
    "cum_flow2",
    "gini_mean",
    "fraction_time_stable",
    "time_to_stability",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# =========================
# CLEAN / FLAGS
# =========================
# Time-to-stability may be blank for some explosive runs
df["time_to_stability"] = pd.to_numeric(df["time_to_stability"], errors="coerce")

# Explosive run definition
df["explosive"] = df["riskS_mean"] > EXPLOSION_THRESHOLD

# Stable run (not explosive)
df["non_explosive"] = ~df["explosive"]

# =========================
# SUMMARY TABLE
# =========================
rows = []

for topology, g in df.groupby("topology", sort=True):
    n_runs = len(g)
    n_explosive = int(g["explosive"].sum())
    explosion_rate = n_explosive / n_runs if n_runs else np.nan

    g_nonexp = g[g["non_explosive"]].copy()

    row = {
        "topology": topology,
        "n_runs": n_runs,
        "n_explosive": n_explosive,
        "explosion_rate": explosion_rate,

        # Across all runs
        "mean_fraction_time_stable_all": g["fraction_time_stable"].mean(),
        "mean_gini_all": g["gini_mean"].mean(),

        # Robust stats on non-explosive runs only
        "median_riskS_mean_nonexpl": g_nonexp["riskS_mean"].median(),
        "median_peak_riskS_nonexpl": g_nonexp["peak_riskS"].median(),
        "median_return_vol_nonexpl": g_nonexp["return_vol"].median(),
        "median_cum_abs_returns_nonexpl": g_nonexp["cum_abs_returns"].median(),
        "median_cum_flow2_nonexpl": g_nonexp["cum_flow2"].median(),
        "median_belief_var_mean_nonexpl": g_nonexp["belief_var_mean"].median(),
        "median_time_to_stability_nonexpl": g_nonexp["time_to_stability"].median(),

        # Optional mean/std on non-explosive runs
        "mean_riskS_mean_nonexpl": g_nonexp["riskS_mean"].mean(),
        "std_riskS_mean_nonexpl": g_nonexp["riskS_mean"].std(),
        "mean_return_vol_nonexpl": g_nonexp["return_vol"].mean(),
        "std_return_vol_nonexpl": g_nonexp["return_vol"].std(),
    }

    rows.append(row)

summary = pd.DataFrame(rows)

# Make the table easier to read
summary = summary.sort_values("topology").reset_index(drop=True)

# Save
summary.to_csv(OUTPUT_CSV, index=False)

# =========================
# PRINT NICE CONSOLE TABLE
# =========================
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("\n=== PAPER TABLE (baseline v2) ===\n")
print(summary.to_string(index=False))

print(f"\nSaved to: {OUTPUT_CSV}")
