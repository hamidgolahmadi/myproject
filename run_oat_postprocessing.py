"""Command-line wrapper for legacy OAT post-processing."""

from src.analysis.oat_postprocessing import run_oat_postprocessing


def main():
    run_oat_postprocessing()


if __name__ == "__main__":
    main()
