"""Merge legacy OAT simulation chunk outputs."""

import glob
import os

import pandas as pd


def merge_oat_chunks(
    indir="oat_results",
    outdir="oat_merged",
):
    """Merge OAT chunk CSV files and save global and per-parameter outputs."""
    os.makedirs(outdir, exist_ok=True)

    pattern = os.path.join(indir, "*_chunk_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No chunk files found in {indir}")

    print(f"Found {len(files)} chunk files.")

    dfs = []
    for file_path in files:
        df = pd.read_csv(file_path)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    all_out = os.path.join(outdir, "oat_all_merged.csv")
    all_df.to_csv(all_out, index=False)
    print(f"Saved merged all-data file: {all_out}")

    for param in sorted(all_df["varied_parameter"].unique()):
        sub = all_df[all_df["varied_parameter"] == param].copy()
        out_path = os.path.join(outdir, f"{param}_merged.csv")
        sub.to_csv(out_path, index=False)
        print(f"Saved per-parameter merged file: {out_path}")

    print("Done.")

    return all_df
