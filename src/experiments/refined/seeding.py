"""Reusable semantic seed derivation for refined experiments.

Seed derivation is a deterministic namespace mapping only.  It is kept
separate from graph generation, shock generation, and market simulation so
structural-only workflows can reproduce the exact graph seeds used by paired
market replications without generating unnecessary non-network randomness.
"""

from __future__ import annotations

import hashlib

import numpy as np


def nonnegative_integer(name: str, value: int) -> int:
    """Validate and normalise a non-negative integer identifier/seed."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def derive_semantic_seed(
    *,
    experiment_seed: int,
    replication_id: int,
    role: str,
    topology_label: str = "",
) -> int:
    """Derive one stable 64-bit seed from semantic identifiers.

    The namespace string is intentionally unchanged from the original paired
    implementation so this refactor preserves all previously generated seeds.
    """

    experiment_seed = nonnegative_integer("experiment_seed", experiment_seed)
    replication_id = nonnegative_integer("replication_id", replication_id)
    if not isinstance(role, str) or role == "" or role != role.strip():
        raise ValueError("role must be a non-empty string without surrounding whitespace")
    if not isinstance(topology_label, str):
        raise TypeError("topology_label must be a string")
    if topology_label != topology_label.strip():
        raise ValueError("topology_label must have no surrounding whitespace")

    payload = (
        f"refined-paired-v1|{experiment_seed}|{replication_id}|"
        f"{role}|{topology_label}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def derive_graph_seed(
    *,
    experiment_seed: int,
    replication_id: int,
    topology_label: str,
) -> int:
    """Return the topology-specific graph seed for one replication."""

    if not isinstance(topology_label, str):
        raise TypeError("topology_label must be a string")
    if topology_label == "" or topology_label != topology_label.strip():
        raise ValueError(
            "topology_label must be non-empty and have no surrounding whitespace"
        )
    return derive_semantic_seed(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        role="graph",
        topology_label=topology_label,
    )
