# merge_oat_chunks.py
# -------------------------------------------------------------
# Merge all OAT chunk CSV files
# -------------------------------------------------------------

import os
import glob
import pandas as pd


def main():
    indir = "oat_results"
    outdir = "oat_merged"
    os.makedirs(outdir, exist_ok=True)

    pattern = os.path.join(indir, "*_chunk_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No chunk files found in {indir}")

    print(f"Found {len(files)} chunk files.")

    # read all
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # save global merged
    all_out = os.path.join(outdir, "oat_all_merged.csv")
    all_df.to_csv(all_out, index=False)
    print(f"Saved merged all-data file: {all_out}")

    # save per-parameter merged files
    for param in sorted(all_df["varied_parameter"].unique()):
        sub = all_df[all_df["varied_parameter"] == param].copy()
        out_path = os.path.join(outdir, f"{param}_merged.csv")
        sub.to_csv(out_path, index=False)
        print(f"Saved per-parameter merged file: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
