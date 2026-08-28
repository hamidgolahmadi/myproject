import os
import pandas as pd
import numpy as np


def main():
    infile = "interaction_merged/interaction_abg_all_merged.csv"
    outdir = "interaction_summary"
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(infile)

    summary = (
        df.groupby(
            ["alpha_social", "beta", "gamma", "topology"],
            as_index=False
        )
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

    outpath = os.path.join(outdir, "interaction_abg_summary.csv")
    summary.to_csv(outpath, index=False)

    print(f"Saved: {outpath}")
    print("Rows:", len(summary))
    print("Topologies:", sorted(summary['topology'].unique().tolist()))
    print("Alpha count:", summary['alpha_social'].nunique())
    print("Beta count:", summary['beta'].nunique())
    print("Gamma count:", summary['gamma'].nunique())


if __name__ == "__main__":
    main()
