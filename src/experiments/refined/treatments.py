"""Construction of report-defined paired topology treatments.

This module combines a previously prepared ``PairedReplicationPlan`` with
explicit topology specifications and common non-network initial conditions.
It generates only topology-specific ``G`` and the neutral graph-supported
``W_0(G)``.  It does not invent or sample non-network initial conditions and
does not run the market simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.model.refined import (
    PeriodShocks,
    RefinedParameters,
    RefinedState,
    initialise_state,
    uniform_attention_from_graph,
    validate_graph_support,
)
from src.topologies.refined import (
    generate_hub_dominated,
    generate_random_fixed_out_degree,
    generate_small_world,
)

from .paired import PairedReplicationPlan


_TOPOLOGY_KINDS = {"random", "small_world", "hub_dominated"}


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_vector(name: str, values: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _finite_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class TopologySpecification:
    """Structural specification for one named benchmark treatment.

    ``kind`` selects one of the three report-defined graph generators.  The
    common link budget ``k`` is explicit.  ``p_sw`` is used only by the
    Small-World benchmark and ``a0`` only by the hub-dominated benchmark.
    """

    topology_label: str
    kind: str
    k: int
    p_sw: float | None = None
    a0: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.topology_label, str):
            raise TypeError("topology_label must be a string")
        if self.topology_label == "" or self.topology_label != self.topology_label.strip():
            raise ValueError("topology_label must be non-empty and have no surrounding whitespace")
        if self.kind not in _TOPOLOGY_KINDS:
            raise ValueError(f"kind must be one of {sorted(_TOPOLOGY_KINDS)}")

        k = _positive_integer("k", self.k)
        object.__setattr__(self, "k", k)

        if self.kind == "random":
            if self.p_sw is not None or self.a0 is not None:
                raise ValueError("random specification does not accept p_sw or a0")
            return

        if self.kind == "small_world":
            if self.p_sw is None:
                raise ValueError("small_world specification requires p_sw")
            if self.a0 is not None:
                raise ValueError("small_world specification does not accept a0")
            p_sw = _finite_scalar("p_sw", self.p_sw)
            if not 0.0 <= p_sw <= 1.0:
                raise ValueError("p_sw must satisfy 0 <= p_sw <= 1")
            if k % 2 != 0:
                raise ValueError("small_world specification requires even k")
            object.__setattr__(self, "p_sw", p_sw)
            return

        if self.a0 is None:
            raise ValueError("hub_dominated specification requires a0")
        if self.p_sw is not None:
            raise ValueError("hub_dominated specification does not accept p_sw")
        a0 = _finite_scalar("a0", self.a0)
        if a0 <= 0.0:
            raise ValueError("a0 must be strictly positive")
        object.__setattr__(self, "a0", a0)


@dataclass(frozen=True, slots=True)
class NonNetworkInitialConditions:
    """Common initial state components excluding topology-dependent ``W_0``."""

    theta: float
    beliefs: np.ndarray
    positions: np.ndarray
    price: float
    reputation: np.ndarray

    def __post_init__(self) -> None:
        theta = _finite_scalar("theta", self.theta)
        price = _finite_scalar("price", self.price)
        beliefs = _finite_vector("beliefs", self.beliefs)
        n = beliefs.size
        if n == 0:
            raise ValueError("initial conditions must contain at least one agent")
        positions = _finite_vector("positions", self.positions, n)
        reputation = _finite_vector("reputation", self.reputation, n)

        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "beliefs", beliefs)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "reputation", reputation)

    @property
    def n_agents(self) -> int:
        return int(self.beliefs.size)


@dataclass(frozen=True, slots=True)
class PreparedTopologyTreatment:
    """One simulation-ready topology treatment within a paired replication."""

    specification: TopologySpecification
    graph_seed: int
    graph: np.ndarray
    initial_state: RefinedState
    shock_path: tuple[PeriodShocks, ...]
    parameters: RefinedParameters

    def __post_init__(self) -> None:
        if not isinstance(self.specification, TopologySpecification):
            raise TypeError("specification must be a TopologySpecification")
        graph_seed = _nonnegative_integer("graph_seed", self.graph_seed)
        if not isinstance(self.initial_state, RefinedState):
            raise TypeError("initial_state must be a RefinedState")
        if not isinstance(self.parameters, RefinedParameters):
            raise TypeError("parameters must be a RefinedParameters")

        graph = validate_graph_support(self.graph)
        self.initial_state.validate_against(graph, self.parameters.x_bar)
        if len(self.shock_path) == 0:
            raise ValueError("shock_path must contain at least one period")
        if not all(isinstance(shock, PeriodShocks) for shock in self.shock_path):
            raise TypeError("shock_path must contain only PeriodShocks objects")
        if any(shock.n_agents != self.initial_state.n_agents for shock in self.shock_path):
            raise ValueError("shock dimension does not match treatment state dimension")

        object.__setattr__(self, "graph_seed", graph_seed)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "shock_path", tuple(self.shock_path))

    @property
    def topology_label(self) -> str:
        return self.specification.topology_label


def _generate_graph(
    specification: TopologySpecification,
    *,
    n_agents: int,
    graph_seed: int,
) -> np.ndarray:
    if specification.kind == "random":
        return generate_random_fixed_out_degree(
            n_agents=n_agents,
            k=specification.k,
            graph_seed=graph_seed,
        )
    if specification.kind == "small_world":
        assert specification.p_sw is not None
        return generate_small_world(
            n_agents=n_agents,
            k=specification.k,
            p_sw=specification.p_sw,
            graph_seed=graph_seed,
        )
    assert specification.a0 is not None
    return generate_hub_dominated(
        n_agents=n_agents,
        k=specification.k,
        a0=specification.a0,
        graph_seed=graph_seed,
    )


def prepare_paired_treatments(
    *,
    plan: PairedReplicationPlan,
    specifications: Iterable[TopologySpecification],
    initial_conditions: NonNetworkInitialConditions,
    parameters: RefinedParameters,
) -> tuple[PreparedTopologyTreatment, ...]:
    """Construct matched simulation-ready treatments without running them.

    The same explicit non-network initial conditions, shock path, and parameter
    object are used for every topology.  For each named treatment only the
    graph seed and graph generator differ.  ``W_0`` is then generated from the
    realised graph using the common neutral rule in Equations (225)-(226).

    Before constructing any graph, the parameter vector, agent dimension, and
    horizon are checked against the values bound into the paired plan when its
    common shock path was generated.  A plan therefore cannot be silently
    reused under a different model specification.
    """

    if not isinstance(plan, PairedReplicationPlan):
        raise TypeError("plan must be a PairedReplicationPlan")
    if not isinstance(initial_conditions, NonNetworkInitialConditions):
        raise TypeError("initial_conditions must be NonNetworkInitialConditions")
    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be a RefinedParameters")

    plan.validate_parameters(parameters)
    if initial_conditions.n_agents != plan.n_agents:
        raise ValueError("common initial-condition dimension does not match paired plan n_agents")
    if len(plan.shock_path) != plan.n_periods:
        raise ValueError("paired plan shock path does not match its bound horizon")

    try:
        specification_tuple = tuple(specifications)
    except TypeError as exc:
        raise TypeError("specifications must be an iterable of TopologySpecification") from exc

    if len(specification_tuple) < 2:
        raise ValueError("paired treatment construction requires at least two specifications")
    if not all(isinstance(spec, TopologySpecification) for spec in specification_tuple):
        raise TypeError("specifications must contain only TopologySpecification objects")

    specification_by_label = {spec.topology_label: spec for spec in specification_tuple}
    if len(specification_by_label) != len(specification_tuple):
        raise ValueError("topology specification labels must be unique")

    plan_labels = set(plan.topology_labels)
    specification_labels = set(specification_by_label)
    if specification_labels != plan_labels:
        missing = sorted(plan_labels - specification_labels)
        extra = sorted(specification_labels - plan_labels)
        raise ValueError(f"specification labels must exactly match plan labels; missing={missing}, extra={extra}")

    common_k = {spec.k for spec in specification_tuple}
    if len(common_k) != 1:
        raise ValueError("all paired benchmark specifications must use the same k")

    n_agents = initial_conditions.n_agents
    if any(shock.n_agents != n_agents for shock in plan.shock_path):
        raise ValueError("plan shock dimension does not match common initial conditions")

    treatments: list[PreparedTopologyTreatment] = []
    for topology_label in plan.topology_labels:
        specification = specification_by_label[topology_label]
        graph_seed = plan.graph_seed_for(topology_label)
        graph = _generate_graph(
            specification,
            n_agents=n_agents,
            graph_seed=graph_seed,
        )
        attention = uniform_attention_from_graph(graph)
        initial_state = initialise_state(
            theta=initial_conditions.theta,
            beliefs=initial_conditions.beliefs,
            positions=initial_conditions.positions,
            price=initial_conditions.price,
            reputation=initial_conditions.reputation,
            attention=attention,
            graph=graph,
            x_bar=parameters.x_bar,
        )

        treatments.append(
            PreparedTopologyTreatment(
                specification=specification,
                graph_seed=graph_seed,
                graph=graph,
                initial_state=initial_state,
                shock_path=plan.shock_path,
                parameters=parameters,
            )
        )

    return tuple(treatments)
