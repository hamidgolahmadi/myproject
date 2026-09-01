"""Semantic seed planning for refined paired topology experiments.

The report-defined confirmatory design uses common random numbers within each
replication.  Shock, initial-state, and later type-assignment randomness are
therefore replication-common, while graph randomness is topology-specific.
This module prepares those inputs without running a Monte Carlo experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np

from src.model.refined.parameters import RefinedParameters
from src.model.refined.shocks import PeriodShocks, generate_shock_path


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _topology_labels(labels: Iterable[str]) -> tuple[str, ...]:
    try:
        values = tuple(labels)
    except TypeError as exc:
        raise TypeError("topology_labels must be an iterable of strings") from exc

    if len(values) < 2:
        raise ValueError("paired design requires at least two topology labels")
    if not all(isinstance(label, str) for label in values):
        raise TypeError("topology_labels must contain only strings")
    if any(label == "" or label != label.strip() for label in values):
        raise ValueError("topology labels must be non-empty and have no surrounding whitespace")
    if len(set(values)) != len(values):
        raise ValueError("topology labels must be unique within a paired replication")
    return values


def _semantic_seed(
    *,
    experiment_seed: int,
    replication_id: int,
    role: str,
    topology_label: str = "",
) -> int:
    """Derive one stable 64-bit seed from semantic identifiers.

    A cryptographic digest is used only as a deterministic namespace mapper;
    it is not an economic or stochastic model assumption.  Including the role
    prevents one source of randomness from sharing a stream with another.
    Including the topology label for graph seeds makes graph-seed assignment
    invariant to the ordering of topology labels in configuration files.
    """

    payload = (
        f"refined-paired-v1|{experiment_seed}|{replication_id}|"
        f"{role}|{topology_label}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


@dataclass(frozen=True, slots=True)
class ReplicationSeeds:
    """Replication-common semantic seeds required by Decisions D025-D026."""

    experiment_seed: int
    replication_id: int
    shock_seed: int
    initial_state_seed: int
    type_assignment_seed: int

    def __post_init__(self) -> None:
        for name in (
            "experiment_seed",
            "replication_id",
            "shock_seed",
            "initial_state_seed",
            "type_assignment_seed",
        ):
            object.__setattr__(self, name, _nonnegative_integer(name, getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PairedReplicationPlan:
    """Common and topology-specific random inputs for one replication.

    ``shock_path`` is generated once and must be reused unchanged across all
    topology treatments.  ``topology_graph_seeds`` contains one independent
    graph seed per named topology treatment.  The initial-state and
    type-assignment seeds are reserved explicitly even though the first
    homogeneous benchmark does not yet use type assignment.
    """

    seeds: ReplicationSeeds
    topology_graph_seeds: tuple[tuple[str, int], ...]
    shock_path: tuple[PeriodShocks, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, ReplicationSeeds):
            raise TypeError("seeds must be a ReplicationSeeds")
        if len(self.topology_graph_seeds) < 2:
            raise ValueError("paired replication must contain at least two topology graph seeds")

        labels: list[str] = []
        normalised: list[tuple[str, int]] = []
        for item in self.topology_graph_seeds:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("topology_graph_seeds entries must be (label, graph_seed) tuples")
            label, graph_seed = item
            if not isinstance(label, str) or label == "" or label != label.strip():
                raise ValueError("topology graph-seed labels must be non-empty strings without surrounding whitespace")
            graph_seed = _nonnegative_integer("graph_seed", graph_seed)
            labels.append(label)
            normalised.append((label, graph_seed))

        if len(set(labels)) != len(labels):
            raise ValueError("topology graph-seed labels must be unique")
        if len(self.shock_path) == 0:
            raise ValueError("shock_path must contain at least one period")
        if not all(isinstance(shock, PeriodShocks) for shock in self.shock_path):
            raise TypeError("shock_path must contain only PeriodShocks objects")

        object.__setattr__(self, "topology_graph_seeds", tuple(normalised))
        object.__setattr__(self, "shock_path", tuple(self.shock_path))

    @property
    def topology_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.topology_graph_seeds)

    def graph_seed_for(self, topology_label: str) -> int:
        """Return the topology-specific graph seed for this replication."""

        for label, graph_seed in self.topology_graph_seeds:
            if label == topology_label:
                return graph_seed
        raise KeyError(f"unknown topology label: {topology_label!r}")


def prepare_paired_replication(
    *,
    experiment_seed: int,
    replication_id: int,
    topology_labels: Iterable[str],
    n_periods: int,
    n_agents: int,
    parameters: RefinedParameters,
) -> PairedReplicationPlan:
    """Prepare semantic seeds and a common shock path for one replication.

    The resulting plan implements the random-input part of the paired design:

    - common across topology treatments: ``shock_seed``, ``initial_state_seed``,
      ``type_assignment_seed``, and the realised ``shock_path``;
    - topology-specific: one ``graph_seed`` per topology label.

    No graph is generated and no simulation is run here.  This keeps treatment
    construction separate from common-random-number preparation.
    """

    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be a RefinedParameters")

    experiment_seed = _nonnegative_integer("experiment_seed", experiment_seed)
    replication_id = _nonnegative_integer("replication_id", replication_id)
    labels = _topology_labels(topology_labels)

    seeds = ReplicationSeeds(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        shock_seed=_semantic_seed(
            experiment_seed=experiment_seed,
            replication_id=replication_id,
            role="shock",
        ),
        initial_state_seed=_semantic_seed(
            experiment_seed=experiment_seed,
            replication_id=replication_id,
            role="initial_state",
        ),
        type_assignment_seed=_semantic_seed(
            experiment_seed=experiment_seed,
            replication_id=replication_id,
            role="type_assignment",
        ),
    )

    topology_graph_seeds = tuple(
        (
            label,
            _semantic_seed(
                experiment_seed=experiment_seed,
                replication_id=replication_id,
                role="graph",
                topology_label=label,
            ),
        )
        for label in labels
    )

    shock_path = generate_shock_path(
        n_periods=n_periods,
        n_agents=n_agents,
        parameters=parameters,
        shock_seed=seeds.shock_seed,
    )

    return PairedReplicationPlan(
        seeds=seeds,
        topology_graph_seeds=topology_graph_seeds,
        shock_path=shock_path,
    )
