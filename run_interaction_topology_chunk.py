"""Command-line wrapper for legacy ABG interaction experiments."""

import argparse

from src.experiments.interaction_runner import (
    run_interaction_chunk,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    run_interaction_chunk(
        samples_csv=args.samples_csv,
        outdir=args.outdir,
        chunk_index=args.chunk_index,
        n_chunks=args.n_chunks,
        topology=args.topology,
        n_seeds=args.n_seeds,
        horizon=args.horizon,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
