"""Refined fixed-topology doctoral market model.

The modules in this package implement the report specification separately
from the legacy/pilot environments.
"""

from .attention import (
    local_reputation_statistics,
    standardised_reputation_scores,
    update_attention,
)
from .beliefs import belief_noise_covariance, update_beliefs
from .fundamentals import (
    fundamental_value,
    private_signals,
    stationary_fundamental_variance,
    update_fundamental,
)
from .market import market_return, price_change, update_price
from .parameters import RefinedParameters
from .reputation import realised_profits, update_reputation
from .shocks import PeriodShocks
from .simulator import SimulationResult, simulate_shock_path
from .state import (
    PeriodOutputs,
    RefinedState,
    build_neighbourhoods,
    initialise_state,
    validate_attention,
    validate_graph_support,
)
from .trading import (
    desired_actions,
    execute_actions,
    inventory_feasible_bounds,
    net_order_flow,
    perceived_values,
    update_positions,
    valuation_gaps,
)
from .transition import transition_one_period

__all__ = [
    "PeriodOutputs",
    "PeriodShocks",
    "RefinedParameters",
    "RefinedState",
    "SimulationResult",
    "belief_noise_covariance",
    "build_neighbourhoods",
    "desired_actions",
    "execute_actions",
    "fundamental_value",
    "initialise_state",
    "inventory_feasible_bounds",
    "local_reputation_statistics",
    "market_return",
    "net_order_flow",
    "perceived_values",
    "price_change",
    "private_signals",
    "realised_profits",
    "simulate_shock_path",
    "standardised_reputation_scores",
    "stationary_fundamental_variance",
    "transition_one_period",
    "update_attention",
    "update_beliefs",
    "update_fundamental",
    "update_positions",
    "update_price",
    "update_reputation",
    "validate_attention",
    "validate_graph_support",
    "valuation_gaps",
]
