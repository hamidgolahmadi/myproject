"""Command-line wrapper for full ABG interaction summarization."""

from src.analysis.interaction_summary import (
    summarize_interaction_results,
)


def main():
    summarize_interaction_results(
        infile="interaction_merged/interaction_abg_all_merged.csv",
        outdir="interaction_summary",
    )


if __name__ == "__main__":
    main()
