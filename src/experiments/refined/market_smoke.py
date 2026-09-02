"""Topology-blind scale and non-degeneracy smoke diagnostics.

This module is a pre-calibration engineering/scientific gate for the
provisional refined baseline.  It runs only a small number of paired R/SW/SF
replications and records absolute scale diagnostics.  The smoke output must not
be used to rank topologies or to estimate confirmatory treatment effects.

The evaluator consumes canonical ``SimulationResult`` objects.  It does not
recompute or alter the model dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.model.refined import (
    SimulationResult,
    local_reputation_statistics,
    simulate_shock_path,
)

from .baseline_specification import (
    RefinedBaselineCandidate,
    first_refined_baseline_candidate,
    generate_neutral_nonnetwork_initial_conditions,
)
from .paired import prepare_paired_replication
from .seeding import nonnegative_integer
from .treatments import prepare_paired_treatments


_ACTION_SATURATION_CUTOFF = 0.99
_NUMERICAL_TOL = 1e-12


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


@dataclass(frozen=True, slots=True)
class MarketScaleSmokeProtocol:
    """Small pre-freeze smoke design, deliberately disjoint from D042 seeds."""

    experiment_seed: int = 2026090203
    n_replications: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_seed",
            nonnegative_integer("experiment_seed", self.experiment_seed),
        )
        object.__setattr__(
            self,
            "n_replications",
            _positive_integer("n_replications", self.n_replications),
        )


@dataclass(frozen=True, slots=True)
class MarketScaleSmokeRecord:
    replication_id: int
    topology_label: str
    return_std: float
    mean_abs_return: float
    max_abs_return: float
    rms_mispricing: float
    max_abs_mispricing: float
    mean_abs_flow_per_agent: float
    rms_flow_per_agent: float
    desired_action_abs_p95: float
    desired_action_saturation_fraction: float
    execution_projection_fraction: float
    inventory_boundary_fraction: float
    median_local_reputation_std: float
    max_local_reputation_std: float
    median_reputation_scale_to_sigma0: float
    mean_attention_mobility: float
    max_attention_mobility: float
    final_attention_distance_from_initial: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replication_id",
            nonnegative_integer("replication_id", self.replication_id),
        )
        if not isinstance(self.topology_label, str) or self.topology_label == "":
            raise ValueError("topology_label must be a non-empty string")

        fraction_names = (
            "desired_action_saturation_fraction",
            "execution_projection_fraction",
            "inventory_boundary_fraction",
        )
        for name in self.__dataclass_fields__:
            if name in {"replication_id", "topology_label"}:
                continue
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if name in fraction_names and value > 1.0:
                raise ValueError(f"{name} must not exceed one")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SmokeMetricSummary:
    metric: str
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class MarketScaleSmokeResult:
    protocol: MarketScaleSmokeProtocol
    candidate: RefinedBaselineCandidate
    records: tuple[MarketScaleSmokeRecord, ...]
    pooled_summary: tuple[SmokeMetricSummary, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, MarketScaleSmokeProtocol):
            raise TypeError("protocol must be MarketScaleSmokeProtocol")
        if not isinstance(self.candidate, RefinedBaselineCandidate):
            raise TypeError("candidate must be RefinedBaselineCandidate")
        expected = self.protocol.n_replications * len(self.candidate.topology_specifications)
        if len(self.records) != expected:
            raise ValueError(f"records must contain exactly {expected} paired-treatment runs")


_METRIC_NAMES = (
    "return_std",
    "mean_abs_return",
    "max_abs_return",
    "rms_mispricing",
    "max_abs_mispricing",
    "mean_abs_flow_per_agent",
    "rms_flow_per_agent",
    "desired_action_abs_p95",
    "desired_action_saturation_fraction",
    "execution_projection_fraction",
    "inventory_boundary_fraction",
    "median_local_reputation_std",
    "max_local_reputation_std",
    "median_reputation_scale_to_sigma0",
    "mean_attention_mobility",
    "max_attention_mobility",
    "final_attention_distance_from_initial",
)


def diagnose_market_scale(
    result: SimulationResult,
    *,
    graph: np.ndarray,
    x_bar: float,
    sigma_0: float,
    replication_id: int,
    topology_label: str,
) -> MarketScaleSmokeRecord:
    """Compute absolute scale diagnostics from one completed simulation."""

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be SimulationResult")
    if result.n_periods < 2:
        raise ValueError("scale diagnostics require at least two simulated periods")
    if not np.isfinite(x_bar) or x_bar <= 0.0:
        raise ValueError("x_bar must be finite and strictly positive")
    if not np.isfinite(sigma_0) or sigma_0 <= 0.0:
        raise ValueError("sigma_0 must be finite and strictly positive")

    outputs = result.period_outputs
    states = result.states[1:]
    n_agents = result.initial_state.n_agents
    if np.asarray(graph).shape != (n_agents, n_agents):
        raise ValueError("graph dimension does not match simulation dimension")

    returns = np.asarray([output.return_ for output in outputs], dtype=float)
    prices = np.asarray([state.price for state in states], dtype=float)
    values = np.asarray([output.fundamental_value for output in outputs], dtype=float)
    flows_per_agent = np.asarray(
        [output.net_order_flow / n_agents for output in outputs],
        dtype=float,
    )
    desired = np.stack([output.desired_actions for output in outputs], axis=0)
    executed = np.stack([output.actions for output in outputs], axis=0)
    positions = np.stack([state.positions for state in states], axis=0)

    local_raw_stds: list[float] = []
    for state in states:
        _, regularised = local_reputation_statistics(
            state.reputation,
            graph,
            sigma_0,
        )
        raw_std = np.sqrt(np.maximum(regularised**2 - sigma_0**2, 0.0))
        local_raw_stds.extend(float(value) for value in raw_std)
    local_raw_std_array = np.asarray(local_raw_stds, dtype=float)

    attention_mobility = []
    for previous, current in zip(result.states[:-1], result.states[1:]):
        attention_mobility.append(
            float(np.linalg.norm(current.attention - previous.attention) / np.sqrt(n_agents))
        )
    attention_mobility_array = np.asarray(attention_mobility, dtype=float)
    final_attention_distance = float(
        np.linalg.norm(result.final_state.attention - result.initial_state.attention)
        / np.sqrt(n_agents)
    )

    mispricing = prices - values
    projection = np.abs(executed - desired) > _NUMERICAL_TOL
    boundary = np.abs(positions) >= float(x_bar) - _NUMERICAL_TOL

    record = MarketScaleSmokeRecord(
        replication_id=replication_id,
        topology_label=topology_label,
        return_std=float(np.std(returns, ddof=1)),
        mean_abs_return=float(np.mean(np.abs(returns))),
        max_abs_return=float(np.max(np.abs(returns))),
        rms_mispricing=float(np.sqrt(np.mean(mispricing**2))),
        max_abs_mispricing=float(np.max(np.abs(mispricing))),
        mean_abs_flow_per_agent=float(np.mean(np.abs(flows_per_agent))),
        rms_flow_per_agent=float(np.sqrt(np.mean(flows_per_agent**2))),
        desired_action_abs_p95=float(np.quantile(np.abs(desired), 0.95)),
        desired_action_saturation_fraction=float(
            np.mean(np.abs(desired) >= _ACTION_SATURATION_CUTOFF)
        ),
        execution_projection_fraction=float(np.mean(projection)),
        inventory_boundary_fraction=float(np.mean(boundary)),
        median_local_reputation_std=float(np.median(local_raw_std_array)),
        max_local_reputation_std=float(np.max(local_raw_std_array)),
        median_reputation_scale_to_sigma0=float(
            np.median(local_raw_std_array) / float(sigma_0)
        ),
        mean_attention_mobility=float(np.mean(attention_mobility_array)),
        max_attention_mobility=float(np.max(attention_mobility_array)),
        final_attention_distance_from_initial=final_attention_distance,
    )

    # Mathematical non-degeneracy only.  Economic acceptability is assessed
    # after viewing the smoke output and is never inferred from topology ranks.
    if record.return_std <= 0.0:
        raise ValueError("smoke path has zero return variation")
    if record.rms_flow_per_agent <= 0.0:
        raise ValueError("smoke path has zero order-flow variation")
    return record


def summarise_market_scale_smoke(
    records: Iterable[MarketScaleSmokeRecord],
) -> tuple[SmokeMetricSummary, ...]:
    """Pool all topology-labelled smoke runs without estimating contrasts."""

    record_tuple = tuple(records)
    if len(record_tuple) == 0:
        raise ValueError("records must not be empty")
    if not all(isinstance(record, MarketScaleSmokeRecord) for record in record_tuple):
        raise TypeError("records must contain only MarketScaleSmokeRecord objects")

    summaries = []
    for metric in _METRIC_NAMES:
        values = np.asarray([getattr(record, metric) for record in record_tuple], dtype=float)
        summaries.append(
            SmokeMetricSummary(
                metric=metric,
                count=int(values.size),
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
            )
        )
    return tuple(summaries)


def run_first_refined_baseline_scale_smoke(
    *,
    protocol: MarketScaleSmokeProtocol | None = None,
    candidate: RefinedBaselineCandidate | None = None,
) -> MarketScaleSmokeResult:
    """Run the small paired pre-freeze scale smoke for the baseline candidate."""

    if protocol is None:
        protocol = MarketScaleSmokeProtocol()
    if candidate is None:
        candidate = first_refined_baseline_candidate()
    if not isinstance(protocol, MarketScaleSmokeProtocol):
        raise TypeError("protocol must be MarketScaleSmokeProtocol")
    if not isinstance(candidate, RefinedBaselineCandidate):
        raise TypeError("candidate must be RefinedBaselineCandidate")

    specifications = candidate.topology_specifications
    labels = tuple(spec.topology_label for spec in specifications)
    records: list[MarketScaleSmokeRecord] = []

    for replication_id in range(protocol.n_replications):
        plan = prepare_paired_replication(
            experiment_seed=protocol.experiment_seed,
            replication_id=replication_id,
            topology_labels=labels,
            n_periods=candidate.horizon,
            n_agents=candidate.n_agents,
            parameters=candidate.parameters,
        )
        initial_conditions = generate_neutral_nonnetwork_initial_conditions(
            n_agents=candidate.n_agents,
            parameters=candidate.parameters,
            initial_state_seed=plan.seeds.initial_state_seed,
        )
        treatments = prepare_paired_treatments(
            plan=plan,
            specifications=specifications,
            initial_conditions=initial_conditions,
            parameters=candidate.parameters,
        )

        for treatment in treatments:
            simulation = simulate_shock_path(
                treatment.initial_state,
                treatment.shock_path,
                treatment.graph,
                treatment.parameters,
                adaptive_attention=True,
            )
            records.append(
                diagnose_market_scale(
                    simulation,
                    graph=treatment.graph,
                    x_bar=treatment.parameters.x_bar,
                    sigma_0=treatment.parameters.sigma_0,
                    replication_id=replication_id,
                    topology_label=treatment.topology_label,
                )
            )

    record_tuple = tuple(records)
    return MarketScaleSmokeResult(
        protocol=protocol,
        candidate=candidate,
        records=record_tuple,
        pooled_summary=summarise_market_scale_smoke(record_tuple),
    )
