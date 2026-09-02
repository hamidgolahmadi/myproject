"""Provisional first refined baseline parameterisation and neutral initialisation.

The doctoral report fixes the model equations and several design rules but does
not provide one complete numerical parameter table for the refined model.  This
module therefore records an explicit *candidate* baseline derived from the
report plus unit-consistent pilot anchors.  It must pass a dedicated scale smoke
run before it can be frozen for calibration or confirmatory topology analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.model.refined import (
    RefinedParameters,
    fundamental_value,
    stationary_fundamental_variance,
)

from .seeding import nonnegative_integer
from .treatments import NonNetworkInitialConditions, TopologySpecification


def _candidate_parameters() -> RefinedParameters:
    """Return the provisional homogeneous refined parameter vector.

    Information-process anchors are inherited from the pilot only where the
    refined equations have the same units.  Price coefficients are converted
    from the pilot's level-price/simple-return convention to the refined
    normalised/log-price-change convention using the pilot reference price 100.
    The inventory limit and fundamental anchor are genuinely new refined design
    choices and therefore require scale validation before freezing.
    """

    return RefinedParameters(
        rho_theta=0.985,
        sigma_theta=0.025,
        v_bar=0.0,
        psi=1.0,
        sigma_s=0.06,
        sigma_b=0.025,
        alpha=0.75,
        kappa=2.4,
        x_bar=5.0,
        chi=0.02,
        lambda_price=0.0002,
        sigma_p=0.001,
        gamma_R=0.9,
        beta=1.0,
        sigma_0=1e-6,
    )


@dataclass(frozen=True, slots=True)
class RefinedBaselineCandidate:
    """Provisional first-stage market specification awaiting scale validation."""

    n_agents: int = 100
    k: int = 6
    horizon: int = 1000
    hub_q: int = 5
    p_sw: float = 0.02
    a0: float = 1.0
    parameters: RefinedParameters = field(default_factory=_candidate_parameters)

    def __post_init__(self) -> None:
        for name in ("n_agents", "k", "horizon", "hub_q"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            value = int(value)
            if value < 1:
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, value)

        if self.k >= self.n_agents:
            raise ValueError("k must be smaller than n_agents")
        if self.k % 2 != 0:
            raise ValueError("k must be even because the matched Small-World benchmark requires it")
        if self.hub_q > self.n_agents:
            raise ValueError("hub_q cannot exceed n_agents")
        if not np.isfinite(self.p_sw) or not 0.0 <= self.p_sw <= 1.0:
            raise ValueError("p_sw must satisfy 0 <= p_sw <= 1")
        if not np.isfinite(self.a0) or self.a0 <= 0.0:
            raise ValueError("a0 must be finite and strictly positive")
        if not isinstance(self.parameters, RefinedParameters):
            raise TypeError("parameters must be RefinedParameters")

        object.__setattr__(self, "p_sw", float(self.p_sw))
        object.__setattr__(self, "a0", float(self.a0))

    @property
    def topology_specifications(self) -> tuple[TopologySpecification, ...]:
        """Return the matched R/SW/SF topology specifications."""

        return (
            TopologySpecification("R", "random", self.k),
            TopologySpecification("SW", "small_world", self.k, p_sw=self.p_sw),
            TopologySpecification("SF", "hub_dominated", self.k, a0=self.a0),
        )

    @property
    def stationary_theta_std(self) -> float:
        return float(np.sqrt(stationary_fundamental_variance(self.parameters)))


def first_refined_baseline_candidate() -> RefinedBaselineCandidate:
    """Return the provisional candidate to be tested before design freeze."""

    return RefinedBaselineCandidate()


def generate_neutral_nonnetwork_initial_conditions(
    *,
    n_agents: int,
    parameters: RefinedParameters,
    initial_state_seed: int,
) -> NonNetworkInitialConditions:
    """Generate the explicit neutral ``X_{0,-W}`` rule for the candidate.

    ``theta_0`` is drawn from the stationary AR(1) distribution.  The initial
    price is placed exactly at contemporaneous fundamental value and all agents
    start with the same state belief ``theta_0``.  Positions and reputations are
    zero.  This removes arbitrary initial mispricing, disagreement, inventory,
    and performance ranking when the evaluation burn-in is zero.
    """

    if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)):
        raise TypeError("n_agents must be an integer")
    n_agents = int(n_agents)
    if n_agents < 1:
        raise ValueError("n_agents must be strictly positive")
    if not isinstance(parameters, RefinedParameters):
        raise TypeError("parameters must be RefinedParameters")
    seed = nonnegative_integer("initial_state_seed", initial_state_seed)

    rng = np.random.default_rng(seed)
    theta_std = np.sqrt(stationary_fundamental_variance(parameters))
    theta_0 = float(rng.normal(loc=0.0, scale=theta_std))

    return NonNetworkInitialConditions(
        theta=theta_0,
        beliefs=np.full(n_agents, theta_0, dtype=float),
        positions=np.zeros(n_agents, dtype=float),
        price=fundamental_value(theta_0, parameters),
        reputation=np.zeros(n_agents, dtype=float),
    )
