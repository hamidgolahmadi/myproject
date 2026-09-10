from dataclasses import replace

import numpy as np
import pytest

from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.gamma_sweep_protocol import (
    FROZEN_GAMMA_GRID,
    GAMMA_SWEEP_ALPHA_ANCHOR,
    GAMMA_SWEEP_BETA_ANCHOR,
    GAMMA_SWEEP_BOOTSTRAP_SEED,
    GAMMA_SWEEP_EXPERIMENT_SEED,
    GammaSweepProtocol,
    first_gamma_sweep_protocol,
)
from src.experiments.refined.paired import prepare_paired_replication


def test_frozen_gamma_sweep_defaults_are_exact():
    protocol = first_gamma_sweep_protocol()
    assert protocol.experiment_seed == 2026091001
    assert protocol.bootstrap_seed == 2026091002
    assert protocol.alpha_anchor == 0.85
    assert protocol.beta_anchor == 5.0
    assert protocol.gamma_grid == (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
    assert protocol.n_replications == 300
    assert protocol.n_bootstrap == 5000
    assert protocol.confidence_level == 0.95
    assert GAMMA_SWEEP_ALPHA_ANCHOR == 0.85
    assert GAMMA_SWEEP_BETA_ANCHOR == 5.0
    assert FROZEN_GAMMA_GRID == protocol.gamma_grid


def test_frozen_gamma_sweep_counts_are_consistent():
    protocol = first_gamma_sweep_protocol()
    assert protocol.n_gamma == 9
    assert protocol.n_matched_blocks == 2700
    assert protocol.n_simulations == 8100


def test_gamma_sweep_seed_namespaces_are_new_and_disjoint():
    assert GAMMA_SWEEP_EXPERIMENT_SEED != GAMMA_SWEEP_BOOTSTRAP_SEED
    prior = {
        2026090201, 2026090202, 2026090203,
        2026090401, 2026090402, 2026090403, 2026090404, 2026090405,
        2026090701, 2026090702,
    }
    assert GAMMA_SWEEP_EXPERIMENT_SEED not in prior
    assert GAMMA_SWEEP_BOOTSTRAP_SEED not in prior


def test_gamma_grid_rejects_one_endpoint():
    with pytest.raises(ValueError, match="0 <= gamma_R < 1"):
        GammaSweepProtocol(gamma_grid=(0.0, 0.9, 0.999, 1.0))


def test_gamma_grid_requires_zero_no_memory_control():
    with pytest.raises(ValueError, match="no-memory"):
        GammaSweepProtocol(gamma_grid=(0.5, 0.9, 0.999))


def test_gamma_grid_requires_d043_anchor():
    with pytest.raises(ValueError, match="0.9 baseline"):
        GammaSweepProtocol(gamma_grid=(0.0, 0.8, 0.99, 0.999))


def test_gamma_grid_requires_high_persistence_stress_point():
    with pytest.raises(ValueError, match="0.999"):
        GammaSweepProtocol(gamma_grid=(0.0, 0.5, 0.9, 0.99))


def test_gamma_sweep_requires_post_d047_beta_anchor():
    with pytest.raises(ValueError, match="beta anchor 5.0"):
        GammaSweepProtocol(beta_anchor=1.0)


def test_gamma_specific_diagnostics_use_relative_effects():
    protocol = first_gamma_sweep_protocol()
    assert protocol.uses_relative_effect("mean_raw_local_reputation_std") is True
    assert protocol.uses_relative_effect("mean_raw_local_reputation_std_over_sigma0") is True


def test_same_replication_preserves_exogenous_crn_across_gamma():
    baseline = first_refined_baseline_specification()
    p0 = replace(baseline.parameters, alpha=0.85, beta=5.0, gamma_R=0.0)
    p1 = replace(baseline.parameters, alpha=0.85, beta=5.0, gamma_R=0.999)
    labels = ("R", "SW", "SF")
    plan0 = prepare_paired_replication(
        experiment_seed=GAMMA_SWEEP_EXPERIMENT_SEED,
        replication_id=23,
        topology_labels=labels,
        n_periods=4,
        n_agents=6,
        parameters=p0,
    )
    plan1 = prepare_paired_replication(
        experiment_seed=GAMMA_SWEEP_EXPERIMENT_SEED,
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
