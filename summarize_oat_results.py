# summarize_oat_results.py
# -------------------------------------------------------------
# Build summaries for each parameter and topology
# -------------------------------------------------------------

import os
import pandas as pd
import numpy as np


def main():
    infile = "oat_merged/oat_all_merged.csv"
    outdir = "oat_summary"
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(infile)

    def extract_varied_value(row):
        return row[row["varied_parameter"]]

    df["varied_value"] = df.apply(extract_varied_value, axis=1)

    # Summary by parameter, topology, varied value
    summary = (
        df.groupby(["varied_parameter", "topology", "varied_value"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            explosion_rate=("exploded", "mean"),
            mean_time_to_explosion=(
                "time_to_explosion",
                lambda x: float(np.mean([v for v in x if v >= 0])) if any(x >= 0) else -1.0
            ),
            mean_final_price=("final_price", "mean"),
            std_final_price=("final_price", "std"),
            mean_return=("mean_return", "mean"),
            std_mean_return=("mean_return", "std"),
            mean_abs_return=("mean_abs_return", "mean"),
            std_abs_return=("mean_abs_return", "std"),
            mean_std_return=("std_return", "mean"),
            std_std_return=("std_return", "std"),
            mean_risk_v=("mean_risk_v", "mean"),
            mean_belief_var=("mean_belief_var", "mean"),
            mean_avg_abs_position=("mean_avg_abs_position", "mean"),
        )
    )

    summary_out = os.path.join(outdir, "oat_summary_by_topology.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Saved: {summary_out}")

    pooled = (
        df.groupby(["varied_parameter", "varied_value"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            explosion_rate=("exploded", "mean"),
            mean_time_to_explosion=(
                "time_to_explosion",
                lambda x: float(np.mean([v for v in x if v >= 0])) if any(x >= 0) else -1.0
            ),
            mean_final_price=("final_price", "mean"),
            std_final_price=("final_price", "std"),
            mean_return=("mean_return", "mean"),
            mean_abs_return=("mean_abs_return", "mean"),
            mean_std_return=("std_return", "mean"),
            mean_risk_v=("mean_risk_v", "mean"),
            mean_belief_var=("mean_belief_var", "mean"),
            mean_avg_abs_position=("mean_avg_abs_position", "mean"),
        )
    )

    pooled_out = os.path.join(outdir, "oat_summary_pooled.csv")
    pooled.to_csv(pooled_out, index=False)
    print(f"Saved: {pooled_out}")

    print("Done.")


if __name__ == "__main__":
    main()
