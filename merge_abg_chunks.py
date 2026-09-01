"""Command-line wrapper for full ABG interaction merging."""

from src.analysis.interaction_merge import (
    merge_interaction_chunks,
)


def main():
    merge_interaction_chunks(
        indir="interaction_results",
        outdir="interaction_merged",
    )


if __name__ == "__main__":
    main()
