"""Refined confirmatory experiment-design utilities.

This package is separate from legacy experiment runners.  It contains only
infrastructure for the report-defined refined paired design.
"""

from .paired import (
    PairedReplicationPlan,
    ReplicationSeeds,
    prepare_paired_replication,
)
from .treatments import (
    NonNetworkInitialConditions,
    PreparedTopologyTreatment,
    TopologySpecification,
    prepare_paired_treatments,
)

__all__ = [
    "NonNetworkInitialConditions",
    "PairedReplicationPlan",
    "PreparedTopologyTreatment",
    "ReplicationSeeds",
    "TopologySpecification",
    "prepare_paired_replication",
    "prepare_paired_treatments",
]
