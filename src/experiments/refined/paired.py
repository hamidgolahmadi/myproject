"""Semantic seed planning for refined paired topology experiments.

The report-defined confirmatory design uses common random numbers within each
replication. Shock, initial-state, and later type-assignment randomness are
therefore replication-common, while graph randomness is topology-specific.
This module prepares those inputs without running a Monte Carlo experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from src.model.refined.parameters import RefinedParameters
from src.model.refined.shocks import PeriodShocks, generate_shock_path

from .seeding import derive_graph_seed, derive_semantic_seed, nonnegative_integer


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


def refined_parameters_fingerprint(parameters: RefinedParameters) -> str:
    """Return a stable SHA-256 fingerprint of a refined parameter vector."""

    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be a RefinedParameters")
    encoded = json.dumps(
        asdict(parameters),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplicationSeeds:
    """Replication-common semantic seeds required by the paired design."""

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
            object.__setattr__(
                self,
                name,
                nonnegative_integer(name, getattr(self, name)),
            )


@dataclass(frozen=True, slots=True)
class PairedReplicationPlan:
    """Common and topology-specific random inputs for one replication.

    ``shock_path`` is generated once and must be reused unchanged across all
    topology treatments. ``topology_graph_seeds`` contains one independent
    graph seed per named topology treatment. The plan is also bound to the
    exact agent dimension, horizon, and refined parameter vector used to
    generate that common shock path.  This prevents a later caller from
    silently reusing shocks under a different economic specification.
    """

    seeds: ReplicationSeeds
    topology_graph_seeds: tuple[tuple[str, int], ...]
    shock_path: tuple[PeriodShocks, ...]
    n_agents: int
    n_periods: int
    parameters_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, ReplicationSeeds):
            raise TypeError("seeds must be a ReplicationSeeds")
        if len(self.topology_graph_seeds) < 2:
            raise ValueError("paired replication must contain at least two topology graph seeds")

        n_agents = nonnegative_integer("n_agents", self.n_agents)
        n_periods = nonnegative_integer("n_periods", self.n_periods)
        if n_agents < 1:
            raise ValueError("n_agents must be strictly positive")
        if n_periods < 1:
            raise ValueError("n_periods must be strictly positive")
        if not isinstance(self.parameters_fingerprint, str) or len(self.parameters_fingerprint) != 64:
            raise ValueError("parameters_fingerprint must be a 64-character SHA-256 hex string")
        try:
            int(self.parameters_fingerprint, 16)
        except ValueError as exc:
            raise ValueError("parameters_fingerprint must contain hexadecimal characters") from exc

        labels: list[str] = []
        normalised: list[tuple[str, int]] = []
        for item in self.topology_graph_seeds:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "topology_graph_seeds entries must be (label, graph_seed) tuples"
                )
            label, graph_seed = item
            if not isinstance(label, str) or label == "" or label != label.strip():
                raise ValueError(
                    "topology graph-seed labels must be non-empty strings without surrounding whitespace"
                )
            graph_seed = nonnegative_integer("graph_seed", graph_seed)
            labels.append(label)
            normalised.append((label, graph_seed))

        if len(set(labels)) != len(labels):
            raise ValueError("topology graph-seed labels must be unique")
        if len(self.shock_path) != n_periods:
            raise ValueError("shock_path length must equal n_periods")
        if not all(isinstance(shock, PeriodShocks) for shock in self.shock_path):
            raise TypeError("shock_path must contain only PeriodShocks objects")
        if any(shock.n_agents != n_agents for shock in self.shock_path):
            raise ValueError("shock_path agent dimension must equal n_agents")

        object.__setattr__(self, "topology_graph_seeds", tuple(normalised))
        object.__setattr__(self, "shock_path", tuple(self.shock_path))
        object.__setattr__(self, "n_agents", n_agents)
        object.__setattr__(self, "n_periods", n_periods)
        object.__setattr__(self, "parameters_fingerprint", self.parameters_fingerprint.lower())

    @property
    def topology_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.topology_graph_seeds)

    def graph_seed_for(self, topology_label: str) -> int:
        """Return the topology-specific graph seed for this replication."""

        for label, graph_seed in self.topology_graph_seeds:
            if label == topology_label:
                return graph_seed
        raise KeyError(f"unknown topology label: {topology_label!r}")

    def validate_parameters(self, parameters: RefinedParameters) -> None:
        """Fail loudly if a caller tries to reuse this plan with other parameters."""

        fingerprint = refined_parameters_fingerprint(parameters)
        if fingerprint != self.parameters_fingerprint:
            raise ValueError(
                "parameters do not match the parameter vector used to generate the paired shock path"
            )


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

    The plan records a fingerprint of the exact parameter vector used for shock
    generation, plus the agent dimension and horizon.  Treatment construction
    validates those bindings before simulation.
    """

    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be a RefinedParameters")

    experiment_seed = nonnegative_integer("experiment_seed", experiment_seed)
    replication_id = nonnegative_integer("replication_id", replication_id)
    labels = _topology_labels(topology_labels)

    seeds = ReplicationSeeds(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        shock_seed=derive_semantic_seed(
            experiment_seed=experiment_seed,
            replication_id=replication_id,
            role="shock",
        ),
        initial_state_seed=derive_semantic_seed(
            experiment_seed=experiment_seed,
            replication_id=replication_id,
            role="initial_state",
        ),
        type_assignment_seed=derive_semantic_seed(
            experiment_seed=experiment_seed,
            replication_id=replication_id,
            role="type_assignment",
        ),
    )

    topology_graph_seeds = tuple(
        (
            label,
            derive_graph_seed(
                experiment_seed=experiment_seed,
                replication_id=replication_id,
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
        n_agents=n_agents,
        n_periods=n_periods,
        parameters_fingerprint=refined_parameters_fingerprint(parameters),
    )
