"""Validated parameter object for the refined fixed-topology market model.

The doctoral report is the scientific source of truth.  This module contains
only the first-stage homogeneous parameters required by Equations (35)-(82).
Agent-level heterogeneity and attention inertia are deliberately deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, slots=True)
class RefinedParameters:
    """Economic parameters for the first-stage refined model.

    The first confirmatory implementation is homogeneous across agents, so
    ``kappa``, ``x_bar``, ``beta`` and ``sigma_s`` are scalars.  Heterogeneous
    counterparts belong to a later extension rather than this baseline object.
    """

    # Fundamentals: Equations (42)-(44)
    rho_theta: float
    sigma_theta: float
    v_bar: float
    psi: float

    # Signals and beliefs: Equations (45), (48)-(50)
    sigma_s: float
    sigma_b: float
    alpha: float

    # Trading and inventory: Equations (66), (68)-(71)
    kappa: float
    x_bar: float

    # Market: Equations (74)-(75)
    chi: float
    lambda_price: float
    sigma_p: float

    # Reputation and attention: Equations (58)-(60), (79)
    gamma_R: float
    beta: float
    sigma_0: float

    def __post_init__(self) -> None:
        """Reject invalid values before a simulation starts."""
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field.name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field.name} must be finite")
            object.__setattr__(self, field.name, float(value))

        if not -1.0 < self.rho_theta < 1.0:
            raise ValueError("rho_theta must satisfy |rho_theta| < 1")
        if self.sigma_theta < 0.0:
            raise ValueError("sigma_theta must be non-negative")
        if self.psi <= 0.0:
            raise ValueError("psi must be strictly positive")

        if self.sigma_s < 0.0:
            raise ValueError("sigma_s must be non-negative")
        if self.sigma_b < 0.0:
            raise ValueError("sigma_b must be non-negative")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")

        if self.kappa <= 0.0:
            raise ValueError("kappa must be strictly positive")
        if self.x_bar <= 0.0:
            raise ValueError("x_bar must be strictly positive")

        if self.chi < 0.0:
            raise ValueError("chi must be non-negative")
        if self.lambda_price < 0.0:
            raise ValueError("lambda_price must be non-negative")
        if self.sigma_p < 0.0:
            raise ValueError("sigma_p must be non-negative")

        if not 0.0 <= self.gamma_R < 1.0:
            raise ValueError("gamma_R must satisfy 0 <= gamma_R < 1")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative in the first-stage selectivity model")
        if self.sigma_0 <= 0.0:
            raise ValueError("sigma_0 must be strictly positive")
