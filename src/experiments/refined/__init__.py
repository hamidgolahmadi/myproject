"""Refined confirmatory experiment-design utilities.

This package is separate from legacy experiment runners and implements the
report-defined refined paired, structural-validation, and evaluation workflows.
"""

from .calibration import (
    StructuralValidationCalibration,
    first_structural_validation_calibration,
)
from .market_metrics import (
    RunLevelMarketOutcomes,
    compute_run_level_market_outcomes,
    maximum_absolute_mispricing,
    mean_absolute_order_flow_per_agent,
    mean_absolute_return,
    return_volatility,
    rms_mispricing,
    time_averaged_belief_variance,
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
    "RunLevelMarketOutcomes",
    "StructuralEnsembleRecord",
    "StructuralEnsembleResult",
    "StructuralValidationCalibration",
    "TopologySpecification",
    "TopologyStructuralSummary",
    "compute_run_level_market_outcomes",
    "derive_graph_seed",
    "derive_semantic_seed",
    "first_structural_validation_calibration",
    "maximum_absolute_mispricing",
    "mean_absolute_order_flow_per_agent",
    "mean_absolute_return",
    "prepare_paired_replication",
    "prepare_paired_treatments",
    "return_volatility",
    "rms_mispricing",
    "run_structural_ensemble",
    "time_averaged_belief_variance",
    "write_structural_result",
]
