# run_interaction_topology_chunk.py

import os
import math
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.experiments.legacy_adaptive_common import (
    FIXED_EXTRA_PARAMS,
    get_chunk_bounds,
    load_topology,
    run_one_sim,
)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--chunk_index", type=int, required=True)
    parser.add_argument("--n_chunks", type=int, required=True)
    parser.add_argument("--topology", type=str, required=True)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--n_workers", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------
    # load sample grid
    # -------------------------
    df = pd.read_csv(args.samples_csv)
    start, end = get_chunk_bounds(len(df), args.chunk_index, args.n_chunks)
    df_chunk = df.iloc[start:end].copy()

    # -------------------------
    # load topology once
    # -------------------------
    P_init = load_topology(args.topology)

    param_cols = [
        "sigma_signal",
        "sigma_belief",
        "rho_y",
        "sigma_y",
        "alpha_social",
        "beta",
        "gamma",
        "trade_sensitivity",
        "price_impact",
        "sigma_price",
    ]

    out_path = os.path.join(
        args.outdir,
        f"interaction_abg_{args.topology}_chunk_{args.chunk_index:04d}.csv"
    )

    buffer = []
    buffer_size = 500

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:

        for _, srow in df_chunk.iterrows():
            sample_id = int(srow["sample_id"])

            env_params = FIXED_EXTRA_PARAMS.copy()
            for p in param_cols:
                env_params[p] = float(srow[p])

            jobs = [
                (P_init, seed, args.horizon, env_params)
                for seed in range(args.n_seeds)
            ]

            futures = {executor.submit(run_one_sim, job): job[1] for job in jobs}

            for future in as_completed(futures):
                seed = futures[future]
                result = future.result()

                row = {
                    "sample_id": sample_id,
                    "topology": args.topology,
                    "seed": seed,
                }

                for p in param_cols:
                    row[p] = float(srow[p])

                row.update(result)
                buffer.append(row)

                if len(buffer) >= buffer_size:
                    pd.DataFrame(buffer).to_csv(
                        out_path,
                        mode="a",
                        header=not os.path.exists(out_path),
                        index=False
                    )
                    buffer = []

    if buffer:
        pd.DataFrame(buffer).to_csv(
            out_path,
            mode="a",
            header=not os.path.exists(out_path),
            index=False
        )

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
