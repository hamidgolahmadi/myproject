import numpy as np
import pytest

from src.experiments.refined.reputation_diagnostics import raw_local_reputation_std


def test_raw_local_reputation_std_uses_graph_neighbourhoods_without_sigma_floor():
    graph = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    reputation = np.array([1.0, 3.0, 5.0])
    result = raw_local_reputation_std(reputation, graph)
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 1.0]))


def test_raw_local_reputation_std_rejects_wrong_shape():
    graph = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        raw_local_reputation_std(np.array([1.0]), graph)
