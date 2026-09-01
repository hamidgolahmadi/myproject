# run_oat_topology_chunk.py

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
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--n_workers", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load data
    df = pd.read_csv(args.samples_csv)
    start, end = get_chunk_bounds(len(df), args.chunk_index, args.n_chunks)
    df_chunk = df.iloc[start:end].copy()

    # load topology once
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

    varied_parameter = str(df_chunk["varied_parameter"].iloc[0])

    out_path = os.path.join(
        args.outdir,
        f"{varied_parameter}_{args.topology}_chunk_{args.chunk_index:04d}.csv"
    )

    buffer = []
    buffer_size = 200

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:

        for _, srow in df_chunk.iterrows():
            sample_id = int(srow["sample_id"])

            env_params = FIXED_EXTRA_PARAMS.copy()
            for p in param_cols:
                env_params[p] = float(srow[p])

            # prepare parallel jobs
            jobs = [
                (P_init, seed, args.horizon, env_params)
                for seed in range(args.n_seeds)
            ]

            futures = [executor.submit(run_one_sim, job) for job in jobs]

            for seed, future in enumerate(as_completed(futures)):
                result = future.result()

                row = {
                    "sample_id": sample_id,
                    "varied_parameter": varied_parameter,
                    "topology": args.topology,
                    "seed": seed,
                }

                for p in param_cols:
                    row[p] = float(srow[p])

                row.update(result)
                buffer.append(row)

                # incremental write
                if len(buffer) >= buffer_size:
                    pd.DataFrame(buffer).to_csv(
                        out_path,
                        mode="a",
                        header=not os.path.exists(out_path),
                        index=False
                    )
                    buffer = []

    # flush remaining
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
