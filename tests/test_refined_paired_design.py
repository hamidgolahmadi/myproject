import numpy as np
import pytest

from src.experiments.refined import prepare_paired_replication
from src.model.refined import RefinedParameters, generate_shock_path


def make_parameters() -> RefinedParameters:
    return RefinedParameters(
        rho_theta=0.8,
        sigma_theta=0.2,
        v_bar=1.0,
        psi=1.1,
        sigma_s=0.3,
        sigma_b=0.1,
        alpha=0.4,
        kappa=1.2,
        x_bar=2.0,
        chi=0.2,
        lambda_price=0.05,
        sigma_p=0.04,
        gamma_R=0.8,
        beta=1.5,
        sigma_0=0.05,
    )


def assert_shock_paths_equal(left, right) -> None:
    assert len(left) == len(right)
    for lhs, rhs in zip(left, right, strict=True):
        assert lhs.u_theta == rhs.u_theta
        np.testing.assert_array_equal(lhs.epsilon_s, rhs.epsilon_s)
        np.testing.assert_array_equal(lhs.epsilon_b, rhs.epsilon_b)
        assert lhs.epsilon_p == rhs.epsilon_p


def make_plan(**overrides):
    kwargs = dict(
        experiment_seed=20260901,
        replication_id=7,
        topology_labels=("random", "small_world", "scale_free"),
        n_periods=5,
        n_agents=4,
        parameters=make_parameters(),
    )
    kwargs.update(overrides)
    return prepare_paired_replication(**kwargs)


def test_prepare_paired_replication_is_exactly_reproducible():
    first = make_plan()
    second = make_plan()

    assert first.seeds == second.seeds
    assert first.topology_graph_seeds == second.topology_graph_seeds
    assert_shock_paths_equal(first.shock_path, second.shock_path)


def test_common_seed_roles_are_semantically_distinct():
    plan = make_plan()
    values = {
        plan.seeds.shock_seed,
        plan.seeds.initial_state_seed,
        plan.seeds.type_assignment_seed,
    }
    assert len(values) == 3


def test_replication_id_changes_common_random_inputs():
    first = make_plan(replication_id=7)
    second = make_plan(replication_id=8)

    assert first.seeds.shock_seed != second.seeds.shock_seed
    assert first.seeds.initial_state_seed != second.seeds.initial_state_seed
    assert first.seeds.type_assignment_seed != second.seeds.type_assignment_seed
    with pytest.raises(AssertionError):
        assert_shock_paths_equal(first.shock_path, second.shock_path)


def test_experiment_seed_changes_common_random_inputs():
    first = make_plan(experiment_seed=11)
    second = make_plan(experiment_seed=12)

    assert first.seeds.shock_seed != second.seeds.shock_seed
    assert first.seeds.initial_state_seed != second.seeds.initial_state_seed
    assert first.seeds.type_assignment_seed != second.seeds.type_assignment_seed


def test_topology_order_does_not_change_named_graph_seeds_or_common_shocks():
    first = make_plan(topology_labels=("random", "small_world", "scale_free"))
    second = make_plan(topology_labels=("scale_free", "random", "small_world"))

    for label in ("random", "small_world", "scale_free"):
        assert first.graph_seed_for(label) == second.graph_seed_for(label)
    assert first.seeds == second.seeds
    assert_shock_paths_equal(first.shock_path, second.shock_path)


def test_different_topology_names_receive_different_graph_seeds():
    plan = make_plan()
    seeds = {plan.graph_seed_for(label) for label in plan.topology_labels}
    assert len(seeds) == len(plan.topology_labels)


def test_adding_topology_does_not_perturb_existing_seed_assignments():
    pair = make_plan(topology_labels=("random", "scale_free"))
    triple = make_plan(topology_labels=("random", "scale_free", "small_world"))

    assert pair.graph_seed_for("random") == triple.graph_seed_for("random")
    assert pair.graph_seed_for("scale_free") == triple.graph_seed_for("scale_free")
    assert pair.seeds == triple.seeds
    assert_shock_paths_equal(pair.shock_path, triple.shock_path)


def test_shock_path_uses_exact_semantic_shock_seed():
    plan = make_plan()
    expected = generate_shock_path(
        n_periods=5,
        n_agents=4,
        parameters=make_parameters(),
        shock_seed=plan.seeds.shock_seed,
    )
    assert_shock_paths_equal(plan.shock_path, expected)


def test_graph_seed_lookup_rejects_unknown_label():
    plan = make_plan()
    with pytest.raises(KeyError, match="unknown topology label"):
        plan.graph_seed_for("not_in_plan")


@pytest.mark.parametrize(
    "labels",
    [
        (),
        ("random",),
        ("random", "random"),
        ("random", " scale_free"),
        ("random", 3),
    ],
)
def test_invalid_topology_label_sets_are_rejected(labels):
    expected = TypeError if any(not isinstance(label, str) for label in labels) else ValueError
    with pytest.raises(expected):
        make_plan(topology_labels=labels)


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("experiment_seed", -1, ValueError),
        ("replication_id", -1, ValueError),
        ("experiment_seed", True, TypeError),
        ("replication_id", 1.5, TypeError),
    ],
)
def test_invalid_semantic_identifiers_are_rejected(field, value, exception):
    with pytest.raises(exception):
        make_plan(**{field: value})


@pytest.mark.parametrize(
    "field,value,exception",
    [
        ("n_periods", 0, ValueError),
        ("n_agents", 0, ValueError),
        ("n_periods", True, TypeError),
        ("n_agents", 2.5, TypeError),
    ],
)
def test_invalid_shock_path_dimensions_are_rejected(field, value, exception):
    with pytest.raises(exception):
        make_plan(**{field: value})


def test_parameters_object_is_required():
    with pytest.raises(TypeError, match="parameters must be a RefinedParameters"):
        make_plan(parameters=None)
