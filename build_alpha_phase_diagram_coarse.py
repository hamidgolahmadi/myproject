"""Command-line wrapper for the coarse ABG alpha phase diagram."""

from src.analysis.interaction_phase_diagram import (
    build_interaction_phase_diagram,
)


def main():
    build_interaction_phase_diagram(
        input_csv=(
            "interaction_summary_coarse/"
            "interaction_abg_summary.csv"
        ),
        outdir="interaction_phase_diagram_coarse",
    )


if __name__ == "__main__":
    main()
