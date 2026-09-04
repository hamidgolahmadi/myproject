from dataclasses import replace

import numpy as np
import pytest

from src.experiments.refined.alpha_sweep_protocol import (
    ALPHA_SWEEP_BOOTSTRAP_SEED,
    ALPHA_SWEEP_EXPERIMENT_SEED,
    FROZEN_ALPHA_GRID,
    AlphaSweepProtocol,
    first_alpha_sweep_protocol,
)
from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.paired import prepare_paired_replication


def test_frozen_alpha_sweep_defaults_are_exact():
    protocol = first_alpha_sweep_protocol()
    assert protocol.experiment_seed == 2026090404
    assert protocol.bootstrap_seed == 2026090405
    assert protocol.alpha_grid == (0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.95, 1.0)
    assert protocol.n_replications == 300
    assert protocol.n_bootstrap == 5000
    assert protocol.confidence_level == 0.95


def test_frozen_alpha_sweep_counts_are_consistent():
    protocol = first_alpha_sweep_protocol()
    assert protocol.n_alpha == 8
    assert protocol.n_matched_blocks == 2400
    assert protocol.n_simulations == 7200


def test_alpha_sweep_seed_namespaces_are_disjoint():
    assert ALPHA_SWEEP_EXPERIMENT_SEED != ALPHA_SWEEP_BOOTSTRAP_SEED
    assert ALPHA_SWEEP_EXPERIMENT_SEED not in {2026090401, 2026090402, 2026090403}
    assert ALPHA_SWEEP_BOOTSTRAP_SEED not in {2026090401, 2026090402, 2026090403}


def test_alpha_grid_rejects_unsorted_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        AlphaSweepProtocol(alpha_grid=(0.0, 0.75, 0.5, 1.0))


def test_alpha_grid_requires_zero_endpoint():
    with pytest.raises(ValueError, match="alpha=0"):
        AlphaSweepProtocol(alpha_grid=(0.1, 0.75, 1.0))


def test_alpha_grid_requires_one_endpoint():
    with pytest.raises(ValueError, match="alpha=1"):
        AlphaSweepProtocol(alpha_grid=(0.0, 0.75, 0.95))


def test_alpha_grid_requires_baseline_anchor():
    with pytest.raises(ValueError, match="0.75"):
        AlphaSweepProtocol(alpha_grid=(0.0, 0.5, 1.0))


def test_alpha_sweep_requires_at_least_1000_bootstrap_draws():
    with pytest.raises(ValueError, match="at least 1000"):
        AlphaSweepProtocol(n_bootstrap=999)


def test_same_replication_preserves_exogenous_crn_across_alpha():
    baseline = first_refined_baseline_specification()
    p0 = replace(baseline.parameters, alpha=0.0)
    p1 = replace(baseline.parameters, alpha=1.0)
    labels = ("R", "SW", "SF")
    plan0 = prepare_paired_replication(
        experiment_seed=ALPHA_SWEEP_EXPERIMENT_SEED,
        replication_id=17,
        topology_labels=labels,
        n_periods=4,
        n_agents=6,
        parameters=p0,
    )
    plan1 = prepare_paired_replication(
        experiment_seed=ALPHA_SWEEP_EXPERIMENT_SEED,
        replication_id=17,
        topology_labels=labels,
        n_periods=4,
        n_agents=6,
        parameters=p1,
    )

    assert plan0.seeds == plan1.seeds
    assert plan0.topology_graph_seeds == plan1.topology_graph_seeds
    assert plan0.parameters_fingerprint != plan1.parameters_fingerprint
    for left, right in zip(plan0.shock_path, plan1.shock_path):
        assert left.u_theta == right.u_theta
        assert left.epsilon_p == right.epsilon_p
        np.testing.assert_array_equal(left.epsilon_s, right.epsilon_s)
        np.testing.assert_array_equal(left.epsilon_b, right.epsilon_b)
