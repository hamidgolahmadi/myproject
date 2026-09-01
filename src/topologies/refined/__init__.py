"""Refined fixed-topology benchmark generators and structural diagnostics."""

from .diagnostics import (
    StructuralDiagnostics,
    average_path_length_lcc,
    diagnose_ensemble,
    diagnose_graph,
    global_clustering,
    hub_link_share,
    in_degree_gini,
    in_degrees,
    largest_component_share,
    symmetrised_support,
)
from .generators import (
    generate_hub_dominated,
    generate_random_fixed_out_degree,
    generate_small_world,
)

__all__ = [
    "StructuralDiagnostics",
    "average_path_length_lcc",
    "diagnose_ensemble",
    "diagnose_graph",
    "generate_hub_dominated",
    "generate_random_fixed_out_degree",
    "generate_small_world",
    "global_clustering",
    "hub_link_share",
    "in_degree_gini",
    "in_degrees",
    "largest_component_share",
    "symmetrised_support",
]
