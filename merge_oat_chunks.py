"""Command-line wrapper for legacy OAT chunk merging."""

from src.analysis.oat_merge import merge_oat_chunks


def main():
    merge_oat_chunks()


if __name__ == "__main__":
    main()
