"""Structural-only ensemble validation for refined benchmark topologies.

This module implements the Section 5.3.1 validation stage before any market
outcome Monte Carlo is interpreted.  It generates matched graph ensembles,
computes report-defined structural diagnostics, preserves graph-level records,
and provides descriptive distribution summaries.

It deliberately does NOT generate shocks, initialise market states, construct
``W_t``, or run the market simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.topologies.refined import StructuralDiagnostics, diagnose_graph

from .seeding import derive_graph_seed, nonnegative_integer
from .treatments import TopologySpecification, _generate_graph


_METRICS = (
    "in_degree_gini",
    "hub_link_share",
    "global_clustering",
    "average_path_length_lcc",
    "largest_component_share",
)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _normalise_specifications(
    specifications: Iterable[TopologySpecification],
) -> tuple[TopologySpecification, ...]:
    try:
        values = tuple(specifications)
    except TypeError as exc:
        raise TypeError("specifications must be an iterable of TopologySpecification") from exc

    if len(values) < 2:
        raise ValueError("structural comparison requires at least two topology specifications")
    if not all(isinstance(spec, TopologySpecification) for spec in values):
        raise TypeError("specifications must contain only TopologySpecification objects")

    labels = [spec.topology_label for spec in values]
    if len(set(labels)) != len(labels):
        raise ValueError("topology specification labels must be unique")

    common_k = {spec.k for spec in values}
    if len(common_k) != 1:
        raise ValueError("all structural benchmark specifications must use the same k")

    return values


@dataclass(frozen=True, slots=True)
class StructuralEnsembleRecord:
    """One graph realisation and its Section 5.3.1 diagnostics."""

    replication_id: int
    topology_label: str
    graph_seed: int
    diagnostics: StructuralDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replication_id",
            nonnegative_integer("replication_id", self.replication_id),
        )
        object.__setattr__(
            self,
            "graph_seed",
            nonnegative_integer("graph_seed", self.graph_seed),
        )
        if not isinstance(self.topology_label, str) or self.topology_label == "":
            raise ValueError("topology_label must be a non-empty string")
        if not isinstance(self.diagnostics, StructuralDiagnostics):
            raise TypeError("diagnostics must be StructuralDiagnostics")


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Descriptive summary that leaves raw graph-level records available."""

    count: int
    mean: float
    std: float
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float


@dataclass(frozen=True, slots=True)
class TopologyStructuralSummary:
    """Distribution summaries for one topology ensemble."""

    topology_label: str
    in_degree_gini: DistributionSummary
    hub_link_share: DistributionSummary
    global_clustering: DistributionSummary
    average_path_length_lcc: DistributionSummary
    largest_component_share: DistributionSummary


@dataclass(frozen=True, slots=True)
class StructuralEnsembleResult:
    """Raw structural-validation records for matched topology ensembles."""

    experiment_seed: int
    n_agents: int
    q: int
    specifications: tuple[TopologySpecification, ...]
    records: tuple[StructuralEnsembleRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_seed",
            nonnegative_integer("experiment_seed", self.experiment_seed),
        )
        object.__setattr__(self, "n_agents", _positive_integer("n_agents", self.n_agents))
        object.__setattr__(self, "q", _positive_integer("q", self.q))
        if self.q > self.n_agents:
            raise ValueError("q must satisfy q <= n_agents")
        object.__setattr__(
            self,
            "specifications",
            _normalise_specifications(self.specifications),
        )
        if len(self.records) == 0:
            raise ValueError("records must contain at least one structural observation")
        if not all(isinstance(record, StructuralEnsembleRecord) for record in self.records):
            raise TypeError("records must contain only StructuralEnsembleRecord objects")
        object.__setattr__(self, "records", tuple(self.records))

    @property
    def topology_labels(self) -> tuple[str, ...]:
        return tuple(spec.topology_label for spec in self.specifications)

    @property
    def n_replications(self) -> int:
        labels = self.topology_labels
        counts = [sum(record.topology_label == label for record in self.records) for label in labels]
        if len(set(counts)) != 1:
            raise RuntimeError("structural result is not balanced across topology labels")
        return counts[0]

    def records_for(self, topology_label: str) -> tuple[StructuralEnsembleRecord, ...]:
        if topology_label not in self.topology_labels:
            raise KeyError(f"unknown topology label: {topology_label!r}")
        return tuple(
            record for record in self.records if record.topology_label == topology_label
        )

    def metric_values(self, topology_label: str, metric: str) -> np.ndarray:
        if metric not in _METRICS:
            raise KeyError(f"unknown structural metric: {metric!r}")
        records = self.records_for(topology_label)
        return np.array(
            [float(getattr(record.diagnostics, metric)) for record in records],
            dtype=float,
        )

    def summary_for(self, topology_label: str) -> TopologyStructuralSummary:
        return TopologyStructuralSummary(
            topology_label=topology_label,
            in_degree_gini=_summarise(self.metric_values(topology_label, "in_degree_gini")),
            hub_link_share=_summarise(self.metric_values(topology_label, "hub_link_share")),
            global_clustering=_summarise(self.metric_values(topology_label, "global_clustering")),
            average_path_length_lcc=_summarise(
                self.metric_values(topology_label, "average_path_length_lcc")
            ),
            largest_component_share=_summarise(
                self.metric_values(topology_label, "largest_component_share")
            ),
        )


def _summarise(values: np.ndarray) -> DistributionSummary:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("summary values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite")

    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return DistributionSummary(
        count=int(array.size),
        mean=float(np.mean(array)),
        std=float(np.std(array, ddof=0)),
        minimum=float(np.min(array)),
        q25=float(q25),
        median=float(median),
        q75=float(q75),
        maximum=float(np.max(array)),
    )


def run_structural_ensemble(
    *,
    experiment_seed: int,
    n_replications: int,
    n_agents: int,
    q: int,
    specifications: Iterable[TopologySpecification],
) -> StructuralEnsembleResult:
    """Generate matched graph ensembles and collect structural diagnostics only.

    Replication ``r`` and topology label ``tau`` use the same deterministic
    graph seed that the paired market design would assign to ``(r, tau)``.
    This makes structural validation directly auditable against later paired
    market replications while avoiding all shock and market-state generation.
    """

    experiment_seed = nonnegative_integer("experiment_seed", experiment_seed)
    n_replications = _positive_integer("n_replications", n_replications)
    n_agents = _positive_integer("n_agents", n_agents)
    q = _positive_integer("q", q)
    if q > n_agents:
        raise ValueError("q must satisfy q <= n_agents")
    specification_tuple = _normalise_specifications(specifications)

    records: list[StructuralEnsembleRecord] = []
    for replication_id in range(n_replications):
        for specification in specification_tuple:
            graph_seed = derive_graph_seed(
                experiment_seed=experiment_seed,
                replication_id=replication_id,
                topology_label=specification.topology_label,
            )
            graph = _generate_graph(
                specification,
                n_agents=n_agents,
                graph_seed=graph_seed,
            )
            diagnostics = diagnose_graph(graph, q=q)
            records.append(
                StructuralEnsembleRecord(
                    replication_id=replication_id,
                    topology_label=specification.topology_label,
                    graph_seed=graph_seed,
                    diagnostics=diagnostics,
                )
            )

    return StructuralEnsembleResult(
        experiment_seed=experiment_seed,
        n_agents=n_agents,
        q=q,
        specifications=specification_tuple,
        records=tuple(records),
    )
