"""Refined confirmatory experiment-design utilities.

This package is separate from legacy experiment runners and implements the
report-defined refined paired and structural-validation workflows.
"""

from .paired import (
    PairedReplicationPlan,
    ReplicationSeeds,
    prepare_paired_replication,
)
from .seeding import derive_graph_seed, derive_semantic_seed
from .structural import (
    DistributionSummary,
    StructuralEnsembleRecord,
    StructuralEnsembleResult,
    TopologyStructuralSummary,
    run_structural_ensemble,
)
from .treatments import (
    NonNetworkInitialConditions,
    PreparedTopologyTreatment,
    TopologySpecification,
    prepare_paired_treatments,
)

__all__ = [
    "DistributionSummary",
    "NonNetworkInitialConditions",
    "PairedReplicationPlan",
    "PreparedTopologyTreatment",
    "ReplicationSeeds",
    "StructuralEnsembleRecord",
    "StructuralEnsembleResult",
    "TopologySpecification",
    "TopologyStructuralSummary",
    "derive_graph_seed",
    "derive_semantic_seed",
    "prepare_paired_replication",
    "prepare_paired_treatments",
    "run_structural_ensemble",
]
