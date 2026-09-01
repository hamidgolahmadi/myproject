"""Command-line wrapper for the full legacy ABG parameter grid."""

from src.experiments.interaction_sampling import (
    generate_full_abg_grid,
)


def main():
    generate_full_abg_grid()


if __name__ == "__main__":
    main()
