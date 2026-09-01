"""Exogenous innovation objects and shock-path generation for the refined model.

Random-number generation is intentionally separate from economic transition
logic. ``PeriodShocks`` stores one already-realised innovation bundle that can
be reused across topology treatments in paired experiments.  The helper
``generate_shock_path`` creates an explicit deterministic path from a semantic
``shock_seed``; transition and simulator code never sample internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import RefinedParameters


def _shock_vector(name: str, value: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _nonnegative_seed(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class PeriodShocks:
    """One period's exogenous innovations.

    Conventions follow Equations (42), (45), (49), and (74):

    - ``u_theta`` is the realised fundamental innovation with variance
      ``sigma_theta**2``;
    - ``epsilon_s`` is the realised private-signal noise vector, already on
      the signal scale;
    - ``epsilon_b`` is the realised belief-processing noise vector, already
      on the belief scale;
    - ``epsilon_p`` is the standard-normal price innovation that is later
      multiplied by ``sigma_p`` in the price equation.

    The object contains no random-number generator and performs no sampling.
    """

    u_theta: float
    epsilon_s: np.ndarray
    epsilon_b: np.ndarray
    epsilon_p: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.u_theta):
            raise ValueError("u_theta must be finite")
        if not np.isfinite(self.epsilon_p):
            raise ValueError("epsilon_p must be finite")

        epsilon_s = _shock_vector("epsilon_s", self.epsilon_s)
        n = epsilon_s.size
        if n == 0:
            raise ValueError("period shocks must contain at least one agent")
        epsilon_b = _shock_vector("epsilon_b", self.epsilon_b, n)

        object.__setattr__(self, "u_theta", float(self.u_theta))
        object.__setattr__(self, "epsilon_s", epsilon_s)
        object.__setattr__(self, "epsilon_b", epsilon_b)
        object.__setattr__(self, "epsilon_p", float(self.epsilon_p))

    @property
    def n_agents(self) -> int:
        return int(self.epsilon_s.size)


def generate_shock_path(
    *,
    n_periods: int,
    n_agents: int,
    parameters: RefinedParameters,
    shock_seed: int,
) -> tuple[PeriodShocks, ...]:
    """Generate a reproducible explicit shock path for paired experiments.

    The four innovation families use independent child streams spawned from
    the semantic ``shock_seed``.  This preserves a stable separation between
    fundamental, signal, belief, and price randomness while retaining a single
    replication-level seed for the exogenous shock path.

    Draws follow the refined baseline assumptions:

    - ``u_theta_t ~ N(0, sigma_theta**2)``;
    - ``epsilon_s_i,t ~ N(0, sigma_s**2)`` independently across agents;
    - ``epsilon_b_i,t ~ N(0, sigma_b**2)`` independently across agents;
    - ``epsilon_p_t ~ N(0, 1)`` and is scaled by ``sigma_p`` only in the price
      transition.

    The returned tuple should be generated once per replication and supplied
    unchanged to every topology treatment in a paired common-random-number
    comparison.
    """

    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be a RefinedParameters")

    n_periods = _positive_integer("n_periods", n_periods)
    n_agents = _positive_integer("n_agents", n_agents)
    shock_seed = _nonnegative_seed("shock_seed", shock_seed)

    root_sequence = np.random.SeedSequence(shock_seed)
    theta_sequence, signal_sequence, belief_sequence, price_sequence = (
        root_sequence.spawn(4)
    )

    theta_rng = np.random.default_rng(theta_sequence)
    signal_rng = np.random.default_rng(signal_sequence)
    belief_rng = np.random.default_rng(belief_sequence)
    price_rng = np.random.default_rng(price_sequence)

    u_theta = theta_rng.normal(
        loc=0.0,
        scale=parameters.sigma_theta,
        size=n_periods,
    )
    epsilon_s = signal_rng.normal(
        loc=0.0,
        scale=parameters.sigma_s,
        size=(n_periods, n_agents),
    )
    epsilon_b = belief_rng.normal(
        loc=0.0,
        scale=parameters.sigma_b,
        size=(n_periods, n_agents),
    )
    epsilon_p = price_rng.standard_normal(size=n_periods)

    return tuple(
        PeriodShocks(
            u_theta=u_theta[t],
            epsilon_s=epsilon_s[t],
            epsilon_b=epsilon_b[t],
            epsilon_p=epsilon_p[t],
        )
        for t in range(n_periods)
    )
