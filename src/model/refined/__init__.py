"""Refined fixed-topology doctoral market model.

The modules in this package implement the report specification separately
from the legacy/pilot environments.
"""

from .fundamentals import (
    fundamental_value,
    private_signals,
    stationary_fundamental_variance,
    update_fundamental,
)
from .parameters import RefinedParameters
from .shocks import PeriodShocks
from .state import (
    PeriodOutputs,
    RefinedState,
    build_neighbourhoods,
    initialise_state,
    validate_attention,
    validate_graph_support,
)

__all__ = [
    "PeriodOutputs",
    "PeriodShocks",
    "RefinedParameters",
    "RefinedState",
    "build_neighbourhoods",
    "fundamental_value",
    "initialise_state",
    "private_signals",
    "stationary_fundamental_variance",
    "update_fundamental",
    "validate_attention",
    "validate_graph_support",
]
