"""Refined fixed-topology doctoral market model.

The modules in this package implement the report specification separately
from the legacy/pilot environments.
"""

from .parameters import RefinedParameters
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
    "RefinedParameters",
    "RefinedState",
    "build_neighbourhoods",
    "initialise_state",
    "validate_attention",
    "validate_graph_support",
]
