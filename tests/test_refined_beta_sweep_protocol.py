from dataclasses import replace

import numpy as np
import pytest

from src.experiments.refined.beta_sweep_protocol import (
    BETA_SWEEP_ALPHA_ANCHOR,
    BETA_SWEEP_BOOTSTRAP_SEED,
    BETA_SWEEP_EXPERIMENT_SEED,
    FROZEN_BETA_GRID,
    BetaSweepProtocol,
    first_beta_sweep_protocol,
)
from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.paired import prepare_paired_replication


def test_frozen_beta_sweep_defaults_are_exact():
    protocol = first_beta_sweep_protocol()
    assert protocol.experiment_seed == 2026090701
    assert protocol.bootstrap_seed == 2026090702
    assert protocol.alpha_anchor == 0.85
    assert protocol.beta_grid == (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0)
    assert protocol.n_replications == 300
    assert protocol.n_bootstrap == 5000
    assert protocol.confidence_level == 0.95
    assert BETA_SWEEP_ALPHA_ANCHOR == 0.85
    assert FROZEN_BETA_GRID == protocol.beta_grid


def test_frozen_beta_sweep_counts_are_consistent():
    protocol = first_beta_sweep_protocol()
    assert protocol.n_beta == 10
    assert protocol.n_matched_blocks == 3000
    assert protocol.n_simulations == 9000


def test_beta_sweep_seed_namespaces_are_disjoint():
    assert BETA_SWEEP_EXPERIMENT_SEED != BETA_SWEEP_BOOTSTRAP_SEED
    prior = {2026090401, 2026090402, 2026090403, 2026090404, 2026090405}
    assert BETA_SWEEP_EXPERIMENT_SEED not in prior
    assert BETA_SWEEP_BOOTSTRAP_SEED not in prior


def test_beta_grid_rejects_unsorted_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        BetaSweepProtocol(beta_grid=(0.0, 1.0, 5.0, 2.0, 1000.0))


def test_beta_grid_requires_zero_control():
    with pytest.raises(ValueError, match="beta=0"):
        BetaSweepProtocol(beta_grid=(0.01, 1.0, 2.0, 5.0, 1000.0))


def test_beta_grid_requires_baseline_anchor():
    with pytest.raises(ValueError, match="beta=1"):
        BetaSweepProtocol(beta_grid=(0.0, 0.1, 2.0, 5.0, 1000.0))


def test_beta_grid_requires_transition_region_resolution():
    with pytest.raises(ValueError, match="beta=2 to 5"):
        BetaSweepProtocol(beta_grid=(0.0, 0.1, 1.0, 10.0, 1000.0))


def test_beta_grid_requires_report_scale_upper_endpoint():
    with pytest.raises(ValueError, match="beta=1000"):
        BetaSweepProtocol(beta_grid=(0.0, 0.1, 1.0, 2.0, 5.0, 100.0))


def test_beta_sweep_requires_at_least_1000_bootstrap_draws():
    with pytest.raises(ValueError, match="at least 1000"):
        BetaSweepProtocol(n_bootstrap=999)


def test_same_replication_preserves_exogenous_crn_across_beta():
    baseline = first_refined_baseline_specification()
    p0 = replace(baseline.parameters, alpha=0.85, beta=0.0)
    p1 = replace(baseline.parameters, alpha=0.85, beta=1000.0)
    labels = ("R", "SW", "SF")
    plan0 = prepare_paired_replication(
        experiment_seed=BETA_SWEEP_EXPERIMENT_SEED,
        replication_id=23,
        topology_labels=labels,
        n_periods=4,
        n_agents=6,
        parameters=p0,
    )
    plan1 = prepare_paired_replication(
        experiment_seed=BETA_SWEEP_EXPERIMENT_SEED,
        replication_id=23,
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
