"""Command-line wrapper for the refined coarse ABG parameter grid."""

from src.experiments.interaction_sampling import (
    generate_coarse_abg_grid,
)


def main():
    generate_coarse_abg_grid()


if __name__ == "__main__":
    main()
