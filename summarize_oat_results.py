"""Command-line wrapper for legacy OAT result summarization."""

from src.analysis.oat_summary import summarize_oat_results


def main():
    summarize_oat_results()


if __name__ == "__main__":
    main()
