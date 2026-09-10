"""Reputation-dispersion diagnostics for the refined fixed-topology model.

These diagnostics are descriptive sidecars. They do not alter the reputation
transition, the attention rule, or any frozen D042--D047 outcome definition.
They are introduced for D048 because changing gamma_R can change the raw scale
of local reputation differences relative to the fixed sigma_0 regularisation
floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.refined import SimulationResult
from src.model.refined.state import build_neighbourhoods, validate_graph_support


@dataclass(frozen=True, slots=True)
class ReputationDispersionDiagnostics:
    """Run-level summary of raw local reputation dispersion."""

    mean_raw_local_reputation_std: float
    mean_raw_local_reputation_std_over_sigma0: float
    n_periods: int
    n_agents: int

    def __post_init__(self) -> None:
        for name in (
            "mean_raw_local_reputation_std",
            "mean_raw_local_reputation_std_over_sigma0",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.n_periods < 1 or self.n_agents < 1:
            raise ValueError("n_periods and n_agents must be positive")


def raw_local_reputation_std(reputation: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Return each agent's unregularised local reputation standard deviation.

    For agent i this is

        sqrt(mean_{j in N_i} (R_j - mean_{k in N_i} R_k)^2).

    Unlike Equation (58), sigma_0 is deliberately not included. The diagnostic
    therefore measures the raw empirical dispersion that sigma_0 regularises.
    """

    graph_array = validate_graph_support(graph)
    reputation_array = np.asarray(reputation, dtype=float)
    n_agents = graph_array.shape[0]
    if reputation_array.shape != (n_agents,):
        raise ValueError(f"reputation must have shape ({n_agents},)")
    if not np.all(np.isfinite(reputation_array)):
        raise ValueError("reputation must contain only finite values")

    neighbourhoods, _ = build_neighbourhoods(graph_array)
    result = np.empty(n_agents, dtype=float)
    for i, neighbours in enumerate(neighbourhoods):
        local = reputation_array[neighbours]
        local_mean = float(np.mean(local))
        result[i] = np.sqrt(float(np.mean((local - local_mean) ** 2)))
    return result


def reputation_dispersion_diagnostics(
    result: SimulationResult,
    graph: np.ndarray,
    *,
    sigma_0: float,
) -> ReputationDispersionDiagnostics:
    """Average raw local reputation dispersion over t=1,...,T and agents.

    D045--D047 attention diagnostics average the full realised influence path.
    D048 follows the same full-path convention here. The ratio to sigma_0 is
    reported explicitly so a high-gamma attenuation can be distinguished from
    a direct persistence effect when the fixed regularisation floor becomes
    large relative to realised reputation dispersion.
    """

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    graph_array = validate_graph_support(graph)
    if result.initial_state.n_agents != graph_array.shape[0]:
        raise ValueError("graph and simulation state dimensions must agree")
    sigma_0 = float(sigma_0)
    if not np.isfinite(sigma_0) or sigma_0 <= 0.0:
        raise ValueError("sigma_0 must be finite and strictly positive")

    period_means: list[float] = []
    for period in range(1, result.n_periods + 1):
        values = raw_local_reputation_std(result.states[period].reputation, graph_array)
        period_means.append(float(np.mean(values)))

    mean_raw = float(np.mean(period_means))
    return ReputationDispersionDiagnostics(
        mean_raw_local_reputation_std=mean_raw,
        mean_raw_local_reputation_std_over_sigma0=mean_raw / sigma_0,
        n_periods=result.n_periods,
        n_agents=graph_array.shape[0],
    )
