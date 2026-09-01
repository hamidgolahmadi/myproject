"""Command-line wrapper for legacy OAT network-importance detection."""

from src.analysis.oat_network_importance import (
    detect_network_importance,
)


def main():
    detect_network_importance()


if __name__ == "__main__":
    main()
