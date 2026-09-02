"""Refined confirmatory experiment-design utilities.

This package is separate from legacy experiment runners and implements the
report-defined refined paired, structural-validation, calibration, and
evaluation workflows.
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
from .influence_metrics import (
    RealisedInfluencePath,
    RealisedInfluencePoint,
    attention_entropy,
    attention_mobility,
    attention_overlap,
    effective_number_of_sources,
    normalised_attention_entropy,
    realised_hub_influence_share,
    realised_influence_hhi,
    realised_influence_path,
    realised_influence_shares,
    structural_hub_nodes,
)
from .market_calibration import (
    MarketEvaluationCalibration,
    MarketEvaluationCalibrationProtocol,
    calibrate_market_evaluation,
    estimate_cid_threshold,
    estimate_reference_scales,
    first_market_evaluation_calibration_protocol,
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
    "MarketEvaluationCalibration",
    "MarketEvaluationCalibrationProtocol",
    "NonNetworkInitialConditions",
    "OperationalStabilisationResult",
    "PairedReplicationPlan",
    "PreparedTopologyTreatment",
    "RealisedInfluencePath",
    "RealisedInfluencePoint",
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
    "attention_entropy",
    "attention_mobility",
    "attention_overlap",
    "calibrate_market_evaluation",
    "classify_cid_path",
    "compute_run_level_market_outcomes",
    "derive_graph_seed",
    "derive_semantic_seed",
    "effective_number_of_sources",
    "estimate_cid_threshold",
    "estimate_reference_scales",
    "first_market_evaluation_calibration_protocol",
    "first_structural_validation_calibration",
    "maximum_absolute_mispricing",
    "mean_absolute_order_flow_per_agent",
    "mean_absolute_return",
    "normalised_attention_entropy",
    "operational_stabilisation",
    "prepare_paired_replication",
    "prepare_paired_treatments",
    "realised_hub_influence_share",
    "realised_influence_hhi",
    "realised_influence_path",
    "realised_influence_shares",
    "return_volatility",
    "rms_mispricing",
    "rolling_action_covariance",
    "rolling_cid",
    "rolling_cid_components",
    "run_structural_ensemble",
    "standardise_cid_components",
    "structural_hub_nodes",
    "threshold_exceedance_rate",
    "time_averaged_belief_variance",
    "write_structural_result",
]
