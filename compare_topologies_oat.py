# compare_topologies_oat.py
# -------------------------------------------------------------
# Compare topology outcomes for each parameter value
# Adds both absolute range and relative gap
# -------------------------------------------------------------

import os
import pandas as pd
import numpy as np


METRICS = [
    "explosion_rate",
    "mean_final_price",
    "mean_return",
    "mean_abs_return",
    "mean_std_return",
    "mean_risk_v",
    "mean_belief_var",
    "mean_avg_abs_position",
]

EPS = 1e-12


def main():
    infile = "oat_summary/oat_summary_by_topology.csv"
    outdir = "oat_topology_comparison"
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(infile)

    rows = []

    grouped = df.groupby(["varied_parameter", "varied_value"])

    for (param, value), sub in grouped:
        row = {
            "varied_parameter": param,
            "varied_value": value,
            "n_topologies": len(sub),
        }

        for metric in METRICS:
            vals = sub[metric].dropna().values

            if len(vals) > 0:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                vmean = float(np.mean(vals))
                vrange = vmax - vmin
                rel_gap = vrange / (abs(vmean) + EPS)

                row[f"{metric}_min"] = vmin
                row[f"{metric}_max"] = vmax
                row[f"{metric}_range"] = vrange
                row[f"{metric}_mean"] = vmean
                row[f"{metric}_relative_gap"] = rel_gap
            else:
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
                row[f"{metric}_range"] = None
                row[f"{metric}_mean"] = None
                row[f"{metric}_relative_gap"] = None

        # identify topologies at extremes for key metrics
        for metric in ["explosion_rate", "mean_abs_return", "mean_belief_var", "mean_avg_abs_position"]:
            if sub[metric].notna().any():
                max_idx = sub[metric].idxmax()
                min_idx = sub[metric].idxmin()
                row[f"{metric}_max_topology"] = sub.loc[max_idx, "topology"]
                row[f"{metric}_min_topology"] = sub.loc[min_idx, "topology"]
            else:
                row[f"{metric}_max_topology"] = None
                row[f"{metric}_min_topology"] = None

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, "topology_gap_by_parameter_value.csv")
    out_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
