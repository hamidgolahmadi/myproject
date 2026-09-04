from dataclasses import replace

import pytest

from src.experiments.refined.paired import (
    prepare_paired_replication,
    refined_parameters_fingerprint,
)
from src.experiments.refined.baseline_specification import (
    first_refined_baseline_specification,
    generate_neutral_nonnetwork_initial_conditions,
)
from src.experiments.refined.treatments import prepare_paired_treatments


def _plan(parameters=None):
    baseline = first_refined_baseline_specification()
    parameters = parameters or baseline.parameters
    return prepare_paired_replication(
        experiment_seed=91001,
        replication_id=3,
        topology_labels=("R", "SW", "SF"),
        n_periods=8,
        n_agents=baseline.n_agents,
        parameters=parameters,
    )


def test_parameter_fingerprint_is_exactly_reproducible():
    parameters = first_refined_baseline_specification().parameters
    assert refined_parameters_fingerprint(parameters) == refined_parameters_fingerprint(parameters)
    assert len(refined_parameters_fingerprint(parameters)) == 64


def test_parameter_fingerprint_changes_when_shock_parameter_changes():
    parameters = first_refined_baseline_specification().parameters
    changed = replace(parameters, sigma_theta=parameters.sigma_theta * 1.1)
    assert refined_parameters_fingerprint(parameters) != refined_parameters_fingerprint(changed)


def test_paired_plan_records_bound_dimension_horizon_and_parameters():
    baseline = first_refined_baseline_specification()
    plan = _plan()
    assert plan.n_agents == baseline.n_agents
    assert plan.n_periods == 8
    assert plan.parameters_fingerprint == refined_parameters_fingerprint(baseline.parameters)
    plan.validate_parameters(baseline.parameters)


def test_treatment_preparation_rejects_parameter_mismatch_with_plan():
    baseline = first_refined_baseline_specification()
    plan = _plan()
    wrong_parameters = replace(baseline.parameters, sigma_s=baseline.parameters.sigma_s * 1.1)
    initial = generate_neutral_nonnetwork_initial_conditions(
        n_agents=baseline.n_agents,
        parameters=baseline.parameters,
        initial_state_seed=plan.seeds.initial_state_seed,
    )
    with pytest.raises(ValueError, match="parameter vector"):
        prepare_paired_treatments(
            plan=plan,
            specifications=baseline.topology_specifications,
            initial_conditions=initial,
            parameters=wrong_parameters,
        )
