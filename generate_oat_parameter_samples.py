"""Command-line wrapper for legacy OAT parameter sampling."""

from src.experiments.oat_sampling import (
    generate_oat_parameter_samples,
)


def main():
    generate_oat_parameter_samples()


if __name__ == "__main__":
    main()
