"""Deterministic tests for refined trading Equations (63)-(72)."""

import numpy as np
import pytest

from src.model.refined import (
    desired_actions,
    execute_actions,
    inventory_feasible_bounds,
    net_order_flow,
    perceived_values,
    update_positions,
    valuation_gaps,
)


def test_perceived_values_equation_63():
    beliefs = np.array([0.2, -0.4, 0.0])
    result = perceived_values(beliefs, v_bar=1.0, psi=1.5)
    assert np.allclose(result, np.array([1.3, 0.4, 1.0]))


def test_valuation_gaps_use_inherited_price_equation_64():
    perceived = np.array([1.3, 0.4, 1.0])
    result = valuation_gaps(perceived, lagged_price=0.9)
    assert np.allclose(result, np.array([0.4, -0.5, 0.1]))


def test_desired_actions_equal_tanh_mapping_equation_66():
    gaps = np.array([0.5, 0.0, -0.25])
    result = desired_actions(gaps, kappa=2.0)
    assert np.allclose(result, np.tanh(np.array([1.0, 0.0, -0.5])))


def test_desired_actions_are_strictly_bounded_by_one():
    result = desired_actions(np.array([100.0, -100.0]), kappa=5.0)
    assert np.all(np.abs(result) <= 1.0)
    assert result[0] > 0.999
    assert result[1] < -0.999


def test_inventory_feasible_bounds_equation_68():
    positions = np.array([0.8, -0.5, 0.0])
    lower, upper = inventory_feasible_bounds(positions, x_bar=1.0)
    assert np.allclose(lower, np.array([-1.8, -0.5, -1.0]))
    assert np.allclose(upper, np.array([0.2, 1.5, 1.0]))


def test_execution_equals_desired_when_inventory_constraint_is_slack():
    desired = np.array([0.2, -0.3, 0.1])
    positions = np.array([0.0, 0.0, 0.0])
    result = execute_actions(desired, positions, x_bar=1.0)
    assert np.allclose(result, desired)


def test_execution_projects_at_both_inventory_boundaries_equations_69_70():
    desired = np.array([0.8, -0.9, 0.4])
    positions = np.array([0.7, -0.6, 0.0])
    result = execute_actions(desired, positions, x_bar=1.0)
    assert np.allclose(result, np.array([0.3, -0.4, 0.4]))


def test_position_update_equation_71():
    lagged_positions = np.array([0.7, -0.6, 0.0])
    actions = np.array([0.3, -0.4, 0.4])
    result = update_positions(lagged_positions, actions)
    assert np.allclose(result, np.array([1.0, -1.0, 0.4]))


def test_net_order_flow_is_signed_not_gross_volume_equation_72():
    actions = np.array([0.6, -0.6, 0.2, -0.1])
    result = net_order_flow(actions)
    assert result == pytest.approx(0.1)
    assert result != pytest.approx(np.sum(np.abs(actions)))


def test_full_deterministic_trading_block_matches_report_sequence():
    beliefs = np.array([0.4, -0.2, 0.1])
    lagged_positions = np.array([0.9, -0.9, 0.0])

    perceived = perceived_values(beliefs, v_bar=1.0, psi=2.0)
    gaps = valuation_gaps(perceived, lagged_price=1.1)
    desired = desired_actions(gaps, kappa=1.5)
    actions = execute_actions(desired, lagged_positions, x_bar=1.0)
    positions = update_positions(lagged_positions, actions)
    flow = net_order_flow(actions)

    expected_perceived = np.array([1.8, 0.6, 1.2])
    expected_gaps = np.array([0.7, -0.5, 0.1])
    expected_desired = np.tanh(1.5 * expected_gaps)
    expected_actions = np.array([0.1, -0.1, expected_desired[2]])

    assert np.allclose(perceived, expected_perceived)
    assert np.allclose(gaps, expected_gaps)
    assert np.allclose(desired, expected_desired)
    assert np.allclose(actions, expected_actions)
    assert np.allclose(positions, lagged_positions + expected_actions)
    assert flow == pytest.approx(float(np.sum(expected_actions)))
    assert np.all(np.abs(positions) <= 1.0 + 1e-12)


def test_trading_block_rejects_invalid_inventory_state_and_parameters():
    with pytest.raises(ValueError):
        perceived_values(np.array([0.0]), v_bar=1.0, psi=0.0)

    with pytest.raises(ValueError):
        desired_actions(np.array([0.0]), kappa=0.0)

    with pytest.raises(ValueError):
        inventory_feasible_bounds(np.array([1.1]), x_bar=1.0)

    with pytest.raises(ValueError):
        execute_actions(np.array([0.1, 0.2]), np.array([0.0]), x_bar=1.0)
