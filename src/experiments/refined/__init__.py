"""Refined confirmatory experiment-design utilities.

This package is separate from legacy experiment runners.  It contains only
infrastructure for the report-defined refined paired design.
"""

from .paired import (
    PairedReplicationPlan,
    ReplicationSeeds,
    prepare_paired_replication,
)

__all__ = [
    "PairedReplicationPlan",
    "ReplicationSeeds",
    "prepare_paired_replication",
]
