import numpy as np
import pytest

from src.experiments.refined import rolling_action_covariance
from src.model.refined import PeriodOutputs, RefinedState, SimulationResult


def _state(n_agents: int) -> RefinedState:
    return RefinedState(
        theta=0.0,
        beliefs=np.zeros(n_agents),
        positions=np.zeros(n_agents),
        price=0.0,
        reputation=np.zeros(n_agents),
        attention=np.eye(n_agents),
    )


def _output(actions: np.ndarray, *, flow: float | None = None) -> PeriodOutputs:
    actions = np.asarray(actions, dtype=float)
    n_agents = actions.size
    net_flow = float(actions.sum()) if flow is None else float(flow)
    return PeriodOutputs(
        fundamental_value=0.0,
        signals=np.zeros(n_agents),
        perceived_values=np.zeros(n_agents),
        valuation_gaps=np.zeros(n_agents),
        desired_actions=actions,
        actions=actions,
        net_order_flow=net_flow,
        return_=0.0,
        profits=np.zeros(n_agents),
        reputation_scores=np.zeros((n_agents, n_agents)),
    )


def _result(actions_by_period: list[list[float]], *, bad_flow_index: int | None = None) -> SimulationResult:
    outputs = []
    for t, actions in enumerate(actions_by_period):
        array = np.asarray(actions, dtype=float)
        flow = None
        if bad_flow_index is not None and t == bad_flow_index:
            flow = float(array.sum() + 1.0)
        outputs.append(_output(array, flow=flow))
    n_agents = len(actions_by_period[0])
    states = tuple(_state(n_agents) for _ in range(len(outputs) + 1))
    return SimulationResult(states=states, period_outputs=tuple(outputs))


ACTIONS = [
    [1.0, 0.0, -1.0],
    [2.0, 1.0, -1.0],
    [3.0, 0.0, -2.0],
    [4.0, 2.0, -1.0],
    [5.0, 1.0, -3.0],
]


def test_endpoint_convention_without_burn_in():
    points = rolling_action_covariance(_result(ACTIONS), window_length=3)
    assert [point.endpoint_period for point in points] == [3, 4, 5]
    assert [point.window_start_period for point in points] == [1, 2, 3]


def test_endpoint_convention_with_burn_in():
    points = rolling_action_covariance(_result(ACTIONS), window_length=2, burn_in=1)
    assert [point.endpoint_period for point in points] == [3, 4, 5]
    assert [point.window_start_period for point in points] == [2, 3, 4]


def test_manual_average_pairwise_covariance_matches_numpy():
    result = _result(ACTIONS)
    point = rolling_action_covariance(result, window_length=3)[0]
    window = np.asarray(ACTIONS[:3], dtype=float)
    covariance = np.cov(window, rowvar=False, ddof=1)
    expected = 2.0 * np.sum(np.triu(covariance, k=1)) / (3 * 2)
    assert point.average_pairwise_action_covariance == pytest.approx(expected)


def test_equation_240_decomposition_holds_for_every_window():
    points = rolling_action_covariance(_result(ACTIONS), window_length=3)
    for point in points:
        assert point.aggregate_order_flow_variance == pytest.approx(
            point.reconstructed_order_flow_variance,
            abs=1e-12,
        )
        assert point.decomposition_error == pytest.approx(0.0, abs=1e-12)


def test_negative_pairwise_covariance_can_cancel_individual_variance():
    result = _result(
        [
            [1.0, -1.0],
            [2.0, -2.0],
            [3.0, -3.0],
        ]
    )
    point = rolling_action_covariance(result, window_length=3)[0]
    assert point.average_pairwise_action_covariance < 0.0
    assert point.sum_individual_action_variances > 0.0
    assert point.aggregate_order_flow_variance == pytest.approx(0.0)
    assert point.reconstructed_order_flow_variance == pytest.approx(0.0)


def test_window_equal_to_post_burn_sample_returns_one_point():
    points = rolling_action_covariance(_result(ACTIONS), window_length=4, burn_in=1)
    assert len(points) == 1
    assert points[0].window_start_period == 2
    assert points[0].endpoint_period == 5


def test_inconsistent_stored_net_flow_is_rejected():
    result = _result(ACTIONS, bad_flow_index=2)
    with pytest.raises(ValueError, match="stored net order flow"):
        rolling_action_covariance(result, window_length=3)


def test_wrong_result_type_is_rejected():
    with pytest.raises(TypeError, match="SimulationResult"):
        rolling_action_covariance(object(), window_length=3)


@pytest.mark.parametrize("window_length", [True, 2.5])
def test_invalid_window_type_is_rejected(window_length):
    with pytest.raises(TypeError, match="window_length"):
        rolling_action_covariance(_result(ACTIONS), window_length=window_length)


@pytest.mark.parametrize("window_length", [0, 1, 6])
def test_invalid_window_value_is_rejected(window_length):
    with pytest.raises(ValueError, match="window_length"):
        rolling_action_covariance(_result(ACTIONS), window_length=window_length)


@pytest.mark.parametrize("burn_in", [True, 1.5])
def test_invalid_burn_in_type_is_rejected(burn_in):
    with pytest.raises(TypeError, match="burn_in"):
        rolling_action_covariance(_result(ACTIONS), window_length=2, burn_in=burn_in)


@pytest.mark.parametrize("burn_in", [-1, 5])
def test_invalid_burn_in_value_is_rejected(burn_in):
    with pytest.raises(ValueError, match="burn_in"):
        rolling_action_covariance(_result(ACTIONS), window_length=2, burn_in=burn_in)


def test_one_agent_result_is_rejected():
    result = _result([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="at least two agents"):
        rolling_action_covariance(result, window_length=2)
