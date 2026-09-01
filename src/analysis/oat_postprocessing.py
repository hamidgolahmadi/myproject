"""Orchestrate the legacy OAT post-processing pipeline."""

from src.analysis.oat_merge import merge_oat_chunks
from src.analysis.oat_network_importance import detect_network_importance
from src.analysis.oat_summary import summarize_oat_results
from src.analysis.oat_topology_comparison import compare_oat_topologies


def run_oat_postprocessing():
    """Run all legacy OAT post-processing stages in their original order."""
    print("\n=== Running OAT merge ===")
    merge_oat_chunks()

    print("\n=== Running OAT summary ===")
    summarize_oat_results()

    print("\n=== Running OAT topology comparison ===")
    compare_oat_topologies()

    print("\n=== Running OAT network-importance detection ===")
    detect_network_importance()

    print("\nAll OAT post-processing steps completed successfully.")
