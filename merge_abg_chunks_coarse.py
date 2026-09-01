"""Command-line wrapper for coarse ABG interaction merging."""

from src.analysis.interaction_merge import (
    merge_interaction_chunks,
)


def main():
    merge_interaction_chunks(
        indir="interaction_results_coarse",
        outdir="interaction_merged_coarse",
    )


if __name__ == "__main__":
    main()
