"""Refined confirmatory experiment-design utilities.

This package is separate from legacy experiment runners and implements the
report-defined refined paired and structural-validation workflows.
"""

from .calibration import (
    StructuralValidationCalibration,
    first_structural_validation_calibration,
)
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
from .structural_io import write_structural_result
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
    "StructuralValidationCalibration",
    "TopologySpecification",
    "TopologyStructuralSummary",
    "derive_graph_seed",
    "derive_semantic_seed",
    "first_structural_validation_calibration",
    "prepare_paired_replication",
    "prepare_paired_treatments",
    "run_structural_ensemble",
    "write_structural_result",
]
