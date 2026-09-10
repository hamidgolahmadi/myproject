from dataclasses import replace

import numpy as np
import pytest

from src.experiments.refined.baseline_specification import first_refined_baseline_specification
from src.experiments.refined.joint_interaction_protocol import (
    FACTORIAL_ALPHA_LEVELS,
    FACTORIAL_BETA_LEVELS,
    FACTORIAL_GAMMA_LEVELS,
    FROZEN_INTERACTION_SPECS,
    FROZEN_JOINT_CELLS,
    JOINT_BOOTSTRAP_SEED,
    JOINT_CORE_OUTCOMES,
    JOINT_EXPERIMENT_SEED,
    JointCell,
    JointInteractionProtocol,
    first_joint_interaction_protocol,
)
from src.experiments.refined.paired import prepare_paired_replication


def test_d049_frozen_defaults_are_exact():
    p = first_joint_interaction_protocol()
    assert p.experiment_seed == 2026091003
    assert p.bootstrap_seed == 2026091004
    assert p.n_replications == 300
    assert p.n_bootstrap == 5000
    assert p.confidence_level == 0.95
    assert p.cells == FROZEN_JOINT_CELLS
    assert p.interaction_specs == FROZEN_INTERACTION_SPECS
    assert p.core_outcomes == JOINT_CORE_OUTCOMES


def test_d049_counts_are_exact():
    p = first_joint_interaction_protocol()
    assert p.n_cells == 50
    assert p.n_checkpoints == 15000
    assert p.n_simulations == 45000


def test_d049_factorial_is_complete_and_reduced():
    p = first_joint_interaction_protocol()
    factorial = [c for c in p.cells if c.cell_kind == "factorial"]
    assert len(factorial) == 48
    assert {(c.alpha, c.beta, c.gamma_R) for c in factorial} == {
        (a, b, g)
        for a in FACTORIAL_ALPHA_LEVELS
        for b in FACTORIAL_BETA_LEVELS
        for g in FACTORIAL_GAMMA_LEVELS
    }


def test_d049_control_and_d043_anchor_are_exact():
    p = first_joint_interaction_protocol()
    alpha0 = next(c for c in p.cells if c.cell_kind == "alpha0_control")
    anchor = next(c for c in p.cells if c.cell_kind == "d043_anchor")
    assert (alpha0.alpha, alpha0.beta, alpha0.gamma_R) == (0.0, 5.0, 0.9)
    assert (anchor.alpha, anchor.beta, anchor.gamma_R) == (0.75, 1.0, 0.9)


def test_d049_seed_namespaces_are_new_and_disjoint():
    prior = {
        2026090201, 2026090202, 2026090203,
        2026090401, 2026090402, 2026090403, 2026090404, 2026090405,
        2026090701, 2026090702, 2026091001, 2026091002,
    }
    assert JOINT_EXPERIMENT_SEED != JOINT_BOOTSTRAP_SEED
    assert JOINT_EXPERIMENT_SEED not in prior
    assert JOINT_BOOTSTRAP_SEED not in prior


def test_d049_rejects_mutated_cell_set():
    p = first_joint_interaction_protocol()
    mutated = list(p.cells)
    mutated[0] = JointCell("F00", "factorial", 0.41, 0.0, 0.0)
    with pytest.raises(ValueError, match="factorial grid"):
        JointInteractionProtocol(cells=tuple(mutated))


def test_d049_interaction_names_are_frozen_and_unique():
    p = first_joint_interaction_protocol()
    names = tuple(spec.name for spec in p.interaction_specs)
    assert len(names) == 9
    assert len(set(names)) == 9
    assert "alpha_beta_gamma_core" in names
    assert "gamma_boundary_shift_at_alpha085_beta5" in names


def test_d049_every_interaction_term_is_a_factorial_cell():
    p = first_joint_interaction_protocol()
    for spec in p.interaction_specs:
        for alpha, beta, gamma_R, _ in spec.terms:
            cell = p.factorial_cell(alpha, beta, gamma_R)
            assert cell.cell_kind == "factorial"


def test_d049_three_way_contrast_has_eight_zero_sum_terms():
    p = first_joint_interaction_protocol()
    spec = next(s for s in p.interaction_specs if s.name == "alpha_beta_gamma_core")
    assert len(spec.terms) == 8
    assert sum(term[3] for term in spec.terms) == pytest.approx(0.0)
    assert set(term[3] for term in spec.terms) == {-1.0, 1.0}


def test_d049_core_outcomes_are_retained():
    p = first_joint_interaction_protocol()
    assert set(JOINT_CORE_OUTCOMES).issubset(p.outcomes)
    assert "mean_raw_local_reputation_std_over_sigma0" in p.outcomes


def test_d049_relative_effect_conventions_are_preserved():
    p = first_joint_interaction_protocol()
    assert p.uses_relative_effect("return_volatility") is True
    assert p.uses_relative_effect("mean_raw_local_reputation_std_over_sigma0") is True
    assert p.uses_relative_effect("threshold_exceeding") is False
    assert p.uses_relative_effect("mean_pairwise_action_covariance") is False


def test_same_replication_preserves_exogenous_crn_across_joint_extremes():
    baseline = first_refined_baseline_specification()
    low = replace(baseline.parameters, alpha=0.4, beta=0.0, gamma_R=0.0)
    high = replace(baseline.parameters, alpha=0.99, beta=100.0, gamma_R=0.999)
    labels = ("R", "SW", "SF")
    plan0 = prepare_paired_replication(
        experiment_seed=JOINT_EXPERIMENT_SEED,
        replication_id=17,
        topology_labels=labels,
        n_periods=4,
        n_agents=6,
        parameters=low,
    )
    plan1 = prepare_paired_replication(
        experiment_seed=JOINT_EXPERIMENT_SEED,
        replication_id=17,
        topology_labels=labels,
        n_periods=4,
        n_agents=6,
        parameters=high,
    )
    assert plan0.seeds == plan1.seeds
    assert plan0.topology_graph_seeds == plan1.topology_graph_seeds
    assert plan0.parameters_fingerprint != plan1.parameters_fingerprint
    for left, right in zip(plan0.shock_path, plan1.shock_path):
        assert left.u_theta == right.u_theta
        assert left.epsilon_p == right.epsilon_p
        np.testing.assert_array_equal(left.epsilon_s, right.epsilon_s)
        np.testing.assert_array_equal(left.epsilon_b, right.epsilon_b)
