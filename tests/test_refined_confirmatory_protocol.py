from dataclasses import replace

import pytest

from src.experiments.refined.confirmatory_protocol import (
    CONFIRMATORY_BOOTSTRAP_SEED,
    CONFIRMATORY_PRODUCTION_SEED,
    CONFIRMATORY_SMOKE_SEED,
    ConfirmatoryProductionProtocol,
    first_confirmatory_production_protocol,
)


def test_d045_defaults_are_frozen_before_production():
    protocol = first_confirmatory_production_protocol()
    assert protocol.experiment_seed == 2026090402
    assert protocol.bootstrap_seed == 2026090403
    assert protocol.n_replications == 1000
    assert protocol.n_bootstrap == 10_000
    assert protocol.confidence_level == pytest.approx(0.95)
    assert protocol.familywise_alpha == pytest.approx(0.05)


def test_production_smoke_and_bootstrap_namespaces_are_disjoint():
    assert len({CONFIRMATORY_SMOKE_SEED, CONFIRMATORY_PRODUCTION_SEED, CONFIRMATORY_BOOTSTRAP_SEED}) == 3


def test_all_three_pairwise_topology_contrasts_are_predeclared():
    protocol = first_confirmatory_production_protocol()
    assert protocol.topology_labels == ("R", "SW", "SF")
    assert protocol.topology_pairs == (("R", "SW"), ("R", "SF"), ("SW", "SF"))


def test_confirmatory_family_sizes_are_predeclared():
    protocol = first_confirmatory_production_protocol()
    assert protocol.primary_family_size == 18
    assert protocol.mechanism_family_size == 12


def test_outcome_families_are_disjoint_and_complete():
    protocol = first_confirmatory_production_protocol()
    groups = [set(protocol.primary_outcomes), set(protocol.mechanism_outcomes), set(protocol.secondary_outcomes)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert len(protocol.all_outcomes) == sum(len(group) for group in groups)


def test_binary_and_signed_outcomes_do_not_use_relative_effects():
    protocol = first_confirmatory_production_protocol()
    assert not protocol.uses_relative_effect("threshold_exceeding")
    assert not protocol.uses_relative_effect("right_censored")
    assert not protocol.uses_relative_effect("mean_pairwise_action_covariance")
    assert protocol.uses_relative_effect("return_volatility")
    assert protocol.uses_relative_effect("mean_attention_overlap")


def test_unknown_relative_effect_outcome_is_rejected():
    with pytest.raises(KeyError):
        first_confirmatory_production_protocol().uses_relative_effect("not_an_outcome")


def test_too_few_production_replications_are_rejected():
    with pytest.raises(ValueError, match="at least two"):
        ConfirmatoryProductionProtocol(n_replications=1)


def test_too_few_bootstrap_draws_are_rejected():
    with pytest.raises(ValueError, match="at least 1000"):
        ConfirmatoryProductionProtocol(n_bootstrap=999)


def test_seed_namespace_collision_is_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        ConfirmatoryProductionProtocol(experiment_seed=CONFIRMATORY_SMOKE_SEED)
    with pytest.raises(ValueError, match="disjoint"):
        ConfirmatoryProductionProtocol(bootstrap_seed=CONFIRMATORY_SMOKE_SEED)


def test_nonstandard_topology_order_is_rejected():
    with pytest.raises(ValueError, match="R/SW/SF"):
        ConfirmatoryProductionProtocol(topology_labels=("SF", "R", "SW"))


def test_invalid_probability_design_inputs_are_rejected():
    with pytest.raises(ValueError):
        ConfirmatoryProductionProtocol(confidence_level=1.0)
    with pytest.raises(ValueError):
        ConfirmatoryProductionProtocol(familywise_alpha=0.0)
    with pytest.raises(ValueError):
        ConfirmatoryProductionProtocol(relative_epsilon=0.0)
