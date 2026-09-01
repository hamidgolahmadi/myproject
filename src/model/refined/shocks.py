"""Exogenous innovation objects for the refined fixed-topology model.

Random-number generation is intentionally separate from economic transition
logic. ``PeriodShocks`` stores one already-realised innovation bundle that can
be reused across topology treatments in paired experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _shock_vector(name: str, value: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


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
