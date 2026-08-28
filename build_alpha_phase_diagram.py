import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "interaction_summary/interaction_abg_summary.csv"
OUTDIR = "interaction_phase_diagram"

RETURN_THRESHOLD = 1.10
BELIEF_THRESHOLD = 0.02

ALPHA_MIN = 0.125
ALPHA_MAX = 0.250
EPS = 1e-12

os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

required_cols = [
    "alpha_social",
    "beta",
    "gamma",
    "topology",
    "mean_abs_return",
    "mean_belief_var",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# فقط بازه موردنظر آلفا
df = df[(df["alpha_social"] >= ALPHA_MIN) & (df["alpha_social"] <= ALPHA_MAX)].copy()

# -----------------------------
# build gaps across topologies
# -----------------------------
group_cols = ["alpha_social", "beta", "gamma"]

gap_rows = []

for keys, g in df.groupby(group_cols):
    alpha_val, beta_val, gamma_val = keys

    ret_vals = g["mean_abs_return"].to_numpy(dtype=float)
    belief_vals = g["mean_belief_var"].to_numpy(dtype=float)

    ret_max = np.max(ret_vals)
    ret_min = np.min(ret_vals)
    ret_mean = np.mean(ret_vals)

    belief_max = np.max(belief_vals)
    belief_min = np.min(belief_vals)
    belief_mean = np.mean(belief_vals)

    ret_range = ret_max - ret_min
    belief_range = belief_max - belief_min

    ret_rel_gap = ret_range / (ret_mean + EPS)
    belief_rel_gap = belief_range / (belief_mean + EPS)

    gap_rows.append({
        "alpha_social": alpha_val,
        "beta": beta_val,
        "gamma": gamma_val,
        "mean_abs_return_range": ret_range,
        "mean_abs_return_relative_gap": ret_rel_gap,
        "mean_belief_var_range": belief_range,
        "mean_belief_var_relative_gap": belief_rel_gap,
    })

gap_df = pd.DataFrame(gap_rows).sort_values(["alpha_social", "beta", "gamma"])

gap_csv = os.path.join(OUTDIR, "interaction_abg_gap_summary.csv")
gap_df.to_csv(gap_csv, index=False)
print(f"Saved: {gap_csv}")

# -----------------------------
# phase diagram by alpha
# -----------------------------
phase_rows = []

for alpha_val, g in gap_df.groupby("alpha_social"):
    total_cells = len(g)

    return_cells = (g["mean_abs_return_relative_gap"] > RETURN_THRESHOLD).sum()
    belief_cells = (g["mean_belief_var_relative_gap"] > BELIEF_THRESHOLD).sum()

    phase_rows.append({
        "alpha_social": alpha_val,
        "total_cells": total_cells,
        "return_cells_above_threshold": int(return_cells),
        "belief_cells_above_threshold": int(belief_cells),
        "return_share": return_cells / total_cells if total_cells > 0 else 0.0,
        "belief_share": belief_cells / total_cells if total_cells > 0 else 0.0,
        "mean_return_relative_gap": g["mean_abs_return_relative_gap"].mean(),
        "mean_belief_relative_gap": g["mean_belief_var_relative_gap"].mean(),
    })

phase_df = pd.DataFrame(phase_rows).sort_values("alpha_social")

phase_csv = os.path.join(OUTDIR, "alpha_phase_diagram_summary.csv")
phase_df.to_csv(phase_csv, index=False)
print(f"Saved: {phase_csv}")

# -----------------------------
# plot 1: shares above threshold
# -----------------------------
plt.figure(figsize=(9, 5))
plt.plot(
    phase_df["alpha_social"],
    phase_df["return_share"],
    marker="o",
    label="Return gap share",
)
plt.plot(
    phase_df["alpha_social"],
    phase_df["belief_share"],
    marker="o",
    label="Belief gap share",
)
plt.xlabel("alpha_social")
plt.ylabel("Share of (beta, gamma) cells above threshold")
plt.title("Topology-relevance phase diagram across alpha")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plot1 = os.path.join(OUTDIR, "alpha_phase_diagram_shares.png")
plt.savefig(plot1, dpi=200)
plt.close()
print(f"Saved: {plot1}")

# -----------------------------
# plot 2: average relative gaps
# -----------------------------
plt.figure(figsize=(9, 5))
plt.plot(
    phase_df["alpha_social"],
    phase_df["mean_return_relative_gap"],
    marker="o",
    label="Mean return relative gap",
)
plt.plot(
    phase_df["alpha_social"],
    phase_df["mean_belief_relative_gap"],
    marker="o",
    label="Mean belief relative gap",
)
plt.xlabel("alpha_social")
plt.ylabel("Average relative gap over (beta, gamma)")
plt.title("Average topology gap across alpha")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plot2 = os.path.join(OUTDIR, "alpha_phase_diagram_mean_gaps.png")
plt.savefig(plot2, dpi=200)
plt.close()
print(f"Saved: {plot2}")
