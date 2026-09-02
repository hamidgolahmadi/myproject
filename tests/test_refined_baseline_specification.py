import numpy as np
import pytest

from src.experiments.refined.baseline_specification import (
    RefinedBaselineCandidate,
    RefinedBaselineSpecification,
    first_refined_baseline_candidate,
    first_refined_baseline_specification,
    generate_neutral_nonnetwork_initial_conditions,
)
from src.experiments.refined.market_calibration import (
    first_market_evaluation_calibration_protocol,
)
from src.model.refined import RefinedParameters, stationary_fundamental_variance


def test_first_specification_has_report_baseline_dimensions():
    specification = first_refined_baseline_specification()
    assert specification.n_agents == 100
    assert specification.k == 6
    assert specification.horizon == 1000
    assert specification.hub_q == 5


def test_first_specification_reuses_validated_structural_calibration():
    specification = first_refined_baseline_specification()
    assert specification.p_sw == pytest.approx(0.02)
    assert specification.a0 == pytest.approx(1.0)


def test_first_specification_parameter_vector_is_explicit_and_frozen():
    p = first_refined_baseline_specification().parameters
    assert p == RefinedParameters(
        rho_theta=0.985,
        sigma_theta=0.025,
        v_bar=0.0,
        psi=1.0,
        sigma_s=0.06,
        sigma_b=0.025,
        alpha=0.75,
        kappa=2.4,
        x_bar=5.0,
        chi=0.02,
        lambda_price=0.0002,
        sigma_p=0.001,
        gamma_R=0.9,
        beta=1.0,
        sigma_0=5e-4,
    )


def test_specification_horizon_matches_market_calibration_protocol():
    assert (
        first_refined_baseline_specification().horizon
        == first_market_evaluation_calibration_protocol().horizon
    )


def test_specification_stationary_theta_std_matches_equation_43():
    specification = first_refined_baseline_specification()
    expected = np.sqrt(stationary_fundamental_variance(specification.parameters))
    assert specification.stationary_theta_std == pytest.approx(expected)


def test_specification_topology_specifications_are_matched_and_ordered():
    specifications = first_refined_baseline_specification().topology_specifications
    assert tuple(spec.topology_label for spec in specifications) == ("R", "SW", "SF")
    assert {spec.k for spec in specifications} == {6}
    assert specifications[1].p_sw == pytest.approx(0.02)
    assert specifications[2].a0 == pytest.approx(1.0)


def test_candidate_class_name_is_compatibility_alias():
    assert RefinedBaselineCandidate is RefinedBaselineSpecification


def test_candidate_factory_returns_frozen_specification():
    assert first_refined_baseline_candidate() == first_refined_baseline_specification()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n_agents": True}, TypeError),
        ({"n_agents": 0}, ValueError),
        ({"k": 100}, ValueError),
        ({"k": 5}, ValueError),
        ({"horizon": 0}, ValueError),
        ({"hub_q": 101}, ValueError),
        ({"p_sw": -0.1}, ValueError),
        ({"p_sw": 1.1}, ValueError),
        ({"a0": 0.0}, ValueError),
        ({"parameters": object()}, TypeError),
    ],
)
def test_specification_rejects_invalid_design_values(kwargs, error):
    with pytest.raises(error):
        RefinedBaselineSpecification(**kwargs)


def _initial_conditions(seed: int = 123):
    specification = first_refined_baseline_specification()
    return generate_neutral_nonnetwork_initial_conditions(
        n_agents=specification.n_agents,
        parameters=specification.parameters,
        initial_state_seed=seed,
    )


def test_neutral_initialisation_is_reproducible():
    first = _initial_conditions(123)
    second = _initial_conditions(123)
    assert first.theta == pytest.approx(second.theta)
    np.testing.assert_array_equal(first.beliefs, second.beliefs)
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.reputation, second.reputation)
    assert first.price == pytest.approx(second.price)


def test_different_initial_state_seeds_change_stationary_theta_draw():
    assert _initial_conditions(123).theta != _initial_conditions(124).theta


def test_neutral_initial_beliefs_equal_theta_for_every_agent():
    initial = _initial_conditions()
    np.testing.assert_allclose(initial.beliefs, initial.theta, atol=0.0, rtol=0.0)


def test_neutral_initial_price_equals_contemporaneous_fundamental_value():
    specification = first_refined_baseline_specification()
    initial = _initial_conditions()
    expected = specification.parameters.v_bar + specification.parameters.psi * initial.theta
    assert initial.price == pytest.approx(expected)


def test_neutral_initial_positions_are_zero():
    np.testing.assert_array_equal(_initial_conditions().positions, np.zeros(100))


def test_neutral_initial_reputations_are_zero():
    np.testing.assert_array_equal(_initial_conditions().reputation, np.zeros(100))


def test_neutral_initialisation_uses_requested_agent_dimension():
    specification = first_refined_baseline_specification()
    initial = generate_neutral_nonnetwork_initial_conditions(
        n_agents=7,
        parameters=specification.parameters,
        initial_state_seed=123,
    )
    assert initial.n_agents == 7


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n_agents": True}, TypeError),
        ({"n_agents": 0}, ValueError),
        ({"parameters": object()}, TypeError),
        ({"initial_state_seed": True}, TypeError),
        ({"initial_state_seed": -1}, ValueError),
    ],
)
def test_neutral_initialisation_rejects_invalid_inputs(kwargs, error):
    specification = first_refined_baseline_specification()
    base = {
        "n_agents": specification.n_agents,
        "parameters": specification.parameters,
        "initial_state_seed": 123,
    }
    base.update(kwargs)
    with pytest.raises(error):
        generate_neutral_nonnetwork_initial_conditions(**base)


def test_neutral_initialisation_does_not_depend_on_topology():
    specification = first_refined_baseline_specification()
    initial = _initial_conditions()
    assert initial.n_agents == specification.n_agents
    # W_0 is intentionally absent: it is created later from each realised G.
    assert not hasattr(initial, "attention")
