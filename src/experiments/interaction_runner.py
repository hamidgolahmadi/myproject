"""Runner for legacy ABG interaction experiments."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from src.experiments.legacy_adaptive_common import (
    FIXED_EXTRA_PARAMS,
    get_chunk_bounds,
    load_topology,
    run_one_sim,
)


PARAM_COLS = [
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


def run_interaction_chunk(
    samples_csv,
    outdir,
    chunk_index,
    n_chunks,
    topology,
    n_seeds=50,
    horizon=3000,
    n_workers=4,
):
    """Run one legacy ABG parameter chunk for one topology."""
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(samples_csv)

    start, end = get_chunk_bounds(
        len(df),
        chunk_index,
        n_chunks,
    )
    df_chunk = df.iloc[start:end].copy()

    P_init = load_topology(topology)

    out_path = os.path.join(
        outdir,
        (
            f"interaction_abg_{topology}_"
            f"chunk_{chunk_index:04d}.csv"
        ),
    )

    buffer = []
    buffer_size = 500

    with ProcessPoolExecutor(
        max_workers=n_workers
    ) as executor:

        for _, srow in df_chunk.iterrows():
            sample_id = int(srow["sample_id"])

            env_params = FIXED_EXTRA_PARAMS.copy()

            for param in PARAM_COLS:
                env_params[param] = float(
                    srow[param]
                )

            jobs = [
                (
                    P_init,
                    seed,
                    horizon,
                    env_params,
                )
                for seed in range(n_seeds)
            ]

            futures = {
                executor.submit(
                    run_one_sim,
                    job,
                ): job[1]
                for job in jobs
            }

            for future in as_completed(futures):
                seed = futures[future]
                result = future.result()

                row = {
                    "sample_id": sample_id,
                    "topology": topology,
                    "seed": seed,
                }

                for param in PARAM_COLS:
                    row[param] = float(
                        srow[param]
                    )

                row.update(result)
                buffer.append(row)

                if len(buffer) >= buffer_size:
                    pd.DataFrame(buffer).to_csv(
                        out_path,
                        mode="a",
                        header=not os.path.exists(
                            out_path
                        ),
                        index=False,
                    )
                    buffer = []

    if buffer:
        pd.DataFrame(buffer).to_csv(
            out_path,
            mode="a",
            header=not os.path.exists(out_path),
            index=False,
        )

    print(f"Saved {out_path}")

    return out_path
