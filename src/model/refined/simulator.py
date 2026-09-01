"""Deterministic shock-path simulation interface for the refined model.

The simulator contains no economic equations of its own.  It advances the
model only by repeatedly calling ``transition_one_period`` on an explicit
sequence of already-realised ``PeriodShocks`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .parameters import RefinedParameters
from .shocks import PeriodShocks
from .state import PeriodOutputs, RefinedState, validate_graph_support
from .transition import transition_one_period


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Persistent states and within-period outputs for a finite shock path.

    ``states`` contains the initial state followed by one state for each
    simulated period, so ``len(states) == n_periods + 1``.  The element
    ``period_outputs[t]`` records the non-persistent objects generated in the
    transition from ``states[t]`` to ``states[t + 1]``.
    """

    states: tuple[RefinedState, ...]
    period_outputs: tuple[PeriodOutputs, ...]

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("states must contain the initial state")
        if len(self.states) != len(self.period_outputs) + 1:
            raise ValueError("states must contain exactly one more element than period_outputs")
        if not all(isinstance(state, RefinedState) for state in self.states):
            raise TypeError("states must contain only RefinedState objects")
        if not all(isinstance(output, PeriodOutputs) for output in self.period_outputs):
            raise TypeError("period_outputs must contain only PeriodOutputs objects")

    @property
    def n_periods(self) -> int:
        return len(self.period_outputs)

    @property
    def initial_state(self) -> RefinedState:
        return self.states[0]

    @property
    def final_state(self) -> RefinedState:
        return self.states[-1]


def simulate_shock_path(
    initial_state: RefinedState,
    shock_path: Iterable[PeriodShocks],
    graph: np.ndarray,
    parameters: RefinedParameters,
    *,
    adaptive_attention: bool = True,
) -> SimulationResult:
    """Simulate an explicit finite shock path by repeated one-period transitions.

    Random-number generation is intentionally outside this function.  Each
    element of ``shock_path`` must already contain the realised innovations for
    one period.  This makes the exact same path reusable across topology
    treatments in paired common-random-number experiments.
    """

    if not isinstance(initial_state, RefinedState):
        raise TypeError("initial_state must be a RefinedState")
    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be a RefinedParameters")
    if not isinstance(adaptive_attention, bool):
        raise TypeError("adaptive_attention must be a bool")

    graph_array = validate_graph_support(graph)
    initial_state.validate_against(graph_array, parameters.x_bar)

    try:
        shocks = tuple(shock_path)
    except TypeError as exc:
        raise TypeError("shock_path must be an iterable of PeriodShocks") from exc

    if len(shocks) == 0:
        raise ValueError("shock_path must contain at least one period")
    if not all(isinstance(shock, PeriodShocks) for shock in shocks):
        raise TypeError("shock_path must contain only PeriodShocks objects")

    states: list[RefinedState] = [initial_state]
    outputs: list[PeriodOutputs] = []
    current_state = initial_state

    for shocks_t in shocks:
        current_state, period_output = transition_one_period(
            current_state,
            shocks_t,
            graph_array,
            parameters,
            adaptive_attention=adaptive_attention,
        )
        states.append(current_state)
        outputs.append(period_output)

    return SimulationResult(states=tuple(states), period_outputs=tuple(outputs))
