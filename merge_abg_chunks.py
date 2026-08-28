# merge_abg_chunks.py

import os
import glob
import pandas as pd


def main():
    indir = "interaction_results"
    outdir = "interaction_merged"
    os.makedirs(outdir, exist_ok=True)

    pattern = os.path.join(indir, "interaction_abg_*_chunk_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No chunk files found in {indir}")

    print(f"Found {len(files)} chunk files.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    outpath = os.path.join(outdir, "interaction_abg_all_merged.csv")
    all_df.to_csv(outpath, index=False)

    print(f"Saved merged file: {outpath}")
    print(f"Total rows: {len(all_df)}")
    print("Topologies found:", sorted(all_df["topology"].unique().tolist()))
    print("Sample count   :", all_df["sample_id"].nunique())


if __name__ == "__main__":
    main()
