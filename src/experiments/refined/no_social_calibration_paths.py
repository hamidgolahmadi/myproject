"""Shared no-social calibration-path construction for D042.

This module contains only experiment orchestration.  It does not reproduce any
economic transition equation: all market dynamics are delegated to the
canonical refined model.  The same helper is used by the small calibration
smoke and the production 500+500 calibration runner.
"""

from __future__ import annotations

from dataclasses import replace

from src.model.refined import (
    generate_shock_path,
    initialise_state,
    simulate_shock_path,
    uniform_attention_from_graph,
)
from src.topologies.refined import generate_random_fixed_out_degree

from .baseline_specification import (
    RefinedBaselineSpecification,
    generate_neutral_nonnetwork_initial_conditions,
)
from .cid import RollingCIDComponentsPoint, rolling_cid_components
from .market_calibration import MarketEvaluationCalibrationProtocol
from .seeding import derive_graph_seed, derive_semantic_seed


_CALIBRATION_GRAPH_LABEL = "CAL"


def no_social_parameters(baseline: RefinedBaselineSpecification):
    """Return the frozen baseline with only social-belief responsiveness removed."""

    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    parameters = replace(baseline.parameters, alpha=0.0)
    if parameters.alpha != 0.0:
        raise RuntimeError("failed to impose the no-social benchmark alpha=0")
    return parameters


def no_social_component_path(
    *,
    experiment_seed: int,
    replication_id: int,
    baseline: RefinedBaselineSpecification,
    protocol: MarketEvaluationCalibrationProtocol,
    adaptive_attention: bool,
) -> tuple[RollingCIDComponentsPoint, ...]:
    """Generate one canonical no-social rolling-component path.

    A single directed Random fixed-out-degree support is generated only because
    the canonical state object requires a valid graph-supported attention
    matrix.  At ``alpha=0`` the graph and attention matrix cannot affect
    beliefs, actions, prices, or the three CID components.

    ``adaptive_attention=False`` is therefore an exact computational shortcut
    for D042 calibration, provided the caller is interested only in the market
    path and CID components.  The equivalence is regression-tested against the
    fully adaptive attention path under identical graph, shock, and initial
    state seeds.
    """

    if not isinstance(baseline, RefinedBaselineSpecification):
        raise TypeError("baseline must be RefinedBaselineSpecification")
    if not isinstance(protocol, MarketEvaluationCalibrationProtocol):
        raise TypeError("protocol must be MarketEvaluationCalibrationProtocol")
    if not isinstance(adaptive_attention, bool):
        raise TypeError("adaptive_attention must be a bool")
    if baseline.horizon != protocol.horizon:
        raise ValueError("baseline horizon must match the calibration protocol horizon")

    parameters = no_social_parameters(baseline)
    shock_seed = derive_semantic_seed(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        role="shock",
    )
    initial_state_seed = derive_semantic_seed(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        role="initial_state",
    )
    graph_seed = derive_graph_seed(
        experiment_seed=experiment_seed,
        replication_id=replication_id,
        topology_label=_CALIBRATION_GRAPH_LABEL,
    )

    graph = generate_random_fixed_out_degree(
        n_agents=baseline.n_agents,
        k=baseline.k,
        graph_seed=graph_seed,
    )
    attention = uniform_attention_from_graph(graph)
    initial = generate_neutral_nonnetwork_initial_conditions(
        n_agents=baseline.n_agents,
        parameters=parameters,
        initial_state_seed=initial_state_seed,
    )
    initial_state = initialise_state(
        theta=initial.theta,
        beliefs=initial.beliefs,
        positions=initial.positions,
        price=initial.price,
        reputation=initial.reputation,
        attention=attention,
        graph=graph,
        x_bar=parameters.x_bar,
    )
    shock_path = generate_shock_path(
        n_periods=protocol.horizon,
        n_agents=baseline.n_agents,
        parameters=parameters,
        shock_seed=shock_seed,
    )
    simulation = simulate_shock_path(
        initial_state,
        shock_path,
        graph,
        parameters,
        adaptive_attention=adaptive_attention,
    )
    components = rolling_cid_components(
        simulation,
        window_length=protocol.rolling_window,
        burn_in=protocol.burn_in,
    )
    if len(components) != protocol.expected_rolling_points_per_run:
        raise RuntimeError("unexpected number of rolling calibration endpoints")
    return components
