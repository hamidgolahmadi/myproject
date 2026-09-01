"""Frozen calibration for the first refined structural-validation run.

The values in this module implement Decision D041.  They are explicit research-
design inputs, not equations of the economic model.  Keeping them in one place
prevents calibration drift between structural validation and later reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .treatments import TopologySpecification


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class StructuralValidationCalibration:
    """Inputs for one matched structural-only benchmark ensemble."""

    experiment_seed: int
    n_agents: int
    k: int
    n_replications: int
    hub_q: int
    p_sw: float
    a0: float

    def __post_init__(self) -> None:
        experiment_seed = _nonnegative_integer("experiment_seed", self.experiment_seed)
        n_agents = _positive_integer("n_agents", self.n_agents)
        k = _positive_integer("k", self.k)
        n_replications = _positive_integer("n_replications", self.n_replications)
        hub_q = _positive_integer("hub_q", self.hub_q)
        p_sw = _finite_scalar("p_sw", self.p_sw)
        a0 = _finite_scalar("a0", self.a0)

        if k > n_agents - 1:
            raise ValueError("k must satisfy k <= n_agents - 1")
        if k % 2 != 0:
            raise ValueError("k must be even for the Small-World benchmark")
        if hub_q > n_agents:
            raise ValueError("hub_q must satisfy hub_q <= n_agents")
        if not 0.0 <= p_sw <= 1.0:
            raise ValueError("p_sw must satisfy 0 <= p_sw <= 1")
        if a0 <= 0.0:
            raise ValueError("a0 must be strictly positive")

        object.__setattr__(self, "experiment_seed", experiment_seed)
        object.__setattr__(self, "n_agents", n_agents)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "n_replications", n_replications)
        object.__setattr__(self, "hub_q", hub_q)
        object.__setattr__(self, "p_sw", p_sw)
        object.__setattr__(self, "a0", a0)

    def topology_specifications(self) -> tuple[TopologySpecification, ...]:
        """Return the matched R/SW/SF structural specifications."""

        return (
            TopologySpecification(topology_label="R", kind="random", k=self.k),
            TopologySpecification(
                topology_label="SW",
                kind="small_world",
                k=self.k,
                p_sw=self.p_sw,
            ),
            TopologySpecification(
                topology_label="SF",
                kind="hub_dominated",
                k=self.k,
                a0=self.a0,
            ),
        )


def first_structural_validation_calibration() -> StructuralValidationCalibration:
    """Return Decision D041 exactly."""

    return StructuralValidationCalibration(
        experiment_seed=20260901,
        n_agents=100,
        k=6,
        n_replications=1000,
        hub_q=5,
        p_sw=0.02,
        a0=1.0,
    )
