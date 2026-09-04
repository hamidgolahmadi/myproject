import numpy as np
import pytest

from src.experiments.refined.action_covariance import rolling_action_covariance
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


def _result(actions: np.ndarray) -> SimulationResult:
    actions = np.asarray(actions, dtype=float)
    n_periods, n_agents = actions.shape
    outputs = tuple(
        PeriodOutputs(
            fundamental_value=0.0,
            signals=np.zeros(n_agents),
            perceived_values=np.zeros(n_agents),
            valuation_gaps=np.zeros(n_agents),
            desired_actions=row,
            actions=row,
            net_order_flow=float(row.sum()),
            return_=0.0,
            profits=np.zeros(n_agents),
            reputation_scores=np.zeros((n_agents, n_agents)),
        )
        for row in actions
    )
    states = tuple(_state(n_agents) for _ in range(n_periods + 1))
    return SimulationResult(states=states, period_outputs=outputs)


def test_vectorised_rolling_covariance_matches_direct_numpy_windows():
    rng = np.random.default_rng(20260904)
    actions = rng.normal(size=(17, 6))
    result = _result(actions)
    window = 5
    burn_in = 2
    points = rolling_action_covariance(result, window_length=window, burn_in=burn_in)

    for offset, point in enumerate(points):
        start = burn_in + offset
        block = actions[start : start + window]
        covariance = np.cov(block, rowvar=False, ddof=1)
        direct_pairwise = 2.0 * np.sum(np.triu(covariance, k=1)) / (6 * 5)
        direct_individual = float(np.trace(covariance))
        direct_flow = float(np.var(block.sum(axis=1), ddof=1))

        assert point.average_pairwise_action_covariance == pytest.approx(direct_pairwise, abs=1e-12)
        assert point.sum_individual_action_variances == pytest.approx(direct_individual, abs=1e-12)
        assert point.aggregate_order_flow_variance == pytest.approx(direct_flow, abs=1e-12)
