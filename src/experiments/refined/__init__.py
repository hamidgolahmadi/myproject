"""Refined confirmatory experiment-design utilities.

This package is separate from legacy experiment runners and implements the
report-defined refined paired, structural-validation, and evaluation workflows.
"""

from .action_covariance import (
    RollingActionCovariancePoint,
    rolling_action_covariance,
)
from .calibration import (
    StructuralValidationCalibration,
    first_structural_validation_calibration,
)
from .cid import (
    CIDReferenceScales,
    CIDWeights,
    RollingCIDComponentsPoint,
    RollingCIDPoint,
    rolling_cid,
    rolling_cid_components,
    standardise_cid_components,
)
from .cid_events import (
    CIDRunClassification,
    CIDThresholdConfiguration,
    OperationalStabilisationResult,
    classify_cid_path,
    operational_stabilisation,
    threshold_exceedance_rate,
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
    "CIDReferenceScales",
    "CIDRunClassification",
    "CIDThresholdConfiguration",
    "CIDWeights",
    "DistributionSummary",
    "NonNetworkInitialConditions",
    "OperationalStabilisationResult",
    "PairedReplicationPlan",
    "PreparedTopologyTreatment",
    "ReplicationSeeds",
    "RollingActionCovariancePoint",
    "RollingCIDComponentsPoint",
    "RollingCIDPoint",
    "RunLevelMarketOutcomes",
    "StructuralEnsembleRecord",
    "StructuralEnsembleResult",
    "StructuralValidationCalibration",
    "TopologySpecification",
    "TopologyStructuralSummary",
    "classify_cid_path",
    "compute_run_level_market_outcomes",
    "derive_graph_seed",
    "derive_semantic_seed",
    "first_structural_validation_calibration",
    "maximum_absolute_mispricing",
    "mean_absolute_order_flow_per_agent",
    "mean_absolute_return",
    "operational_stabilisation",
    "prepare_paired_replication",
    "prepare_paired_treatments",
    "return_volatility",
    "rms_mispricing",
    "rolling_action_covariance",
    "rolling_cid",
    "rolling_cid_components",
    "run_structural_ensemble",
    "standardise_cid_components",
    "threshold_exceedance_rate",
    "time_averaged_belief_variance",
    "write_structural_result",
]
