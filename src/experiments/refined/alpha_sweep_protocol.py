"""Frozen D046 exploratory alpha-sweep protocol.

D046 is an OAT diagnostic following the completed D045 fixed-topology
confirmatory experiment. It maps topology differentiation across the social
weight ``alpha`` while keeping every other D043 parameter and the D044 market
calibration fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .confirmatory_protocol import first_confirmatory_production_protocol


ALPHA_SWEEP_EXPERIMENT_SEED = 2026090404
ALPHA_SWEEP_BOOTSTRAP_SEED = 2026090405
FROZEN_ALPHA_GRID = (0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.95, 1.0)


def _d045_outcomes() -> tuple[str, ...]:
    return first_confirmatory_production_protocol().all_outcomes


@dataclass(frozen=True, slots=True)
class AlphaSweepProtocol:
    """Exploratory matched-block design for D046."""

    experiment_seed: int = ALPHA_SWEEP_EXPERIMENT_SEED
    alpha_grid: tuple[float, ...] = FROZEN_ALPHA_GRID
    n_replications: int = 300
    bootstrap_seed: int = ALPHA_SWEEP_BOOTSTRAP_SEED
    n_bootstrap: int = 5_000
    confidence_level: float = 0.95
    relative_epsilon: float = 1e-12
    topology_labels: tuple[str, ...] = ("R", "SW", "SF")
    topology_pairs: tuple[tuple[str, str], ...] = (
        ("R", "SW"),
        ("R", "SF"),
        ("SW", "SF"),
    )
    outcomes: tuple[str, ...] = field(default_factory=_d045_outcomes)

    def __post_init__(self) -> None:
        for name in ("experiment_seed", "n_replications", "bootstrap_seed", "n_bootstrap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

        if self.n_replications < 2:
            raise ValueError("n_replications must be at least two")
        if self.n_bootstrap < 1000:
            raise ValueError("n_bootstrap must be at least 1000")
        if self.experiment_seed == self.bootstrap_seed:
            raise ValueError("experiment and bootstrap seed namespaces must be disjoint")

        try:
            alpha_grid = tuple(float(value) for value in self.alpha_grid)
        except (TypeError, ValueError) as exc:
            raise TypeError("alpha_grid must contain real scalars") from exc
        if len(alpha_grid) < 2:
            raise ValueError("alpha_grid must contain at least two values")
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in alpha_grid):
            raise ValueError("every alpha value must lie in [0,1]")
        if len(set(alpha_grid)) != len(alpha_grid):
            raise ValueError("alpha_grid values must be unique")
        if tuple(sorted(alpha_grid)) != alpha_grid:
            raise ValueError("alpha_grid must be strictly increasing")
        if alpha_grid[0] != 0.0:
            raise ValueError("D046 alpha_grid must include alpha=0 as its first endpoint")
        if alpha_grid[-1] != 1.0:
            raise ValueError("D046 alpha_grid must include alpha=1 as its final endpoint")
        if 0.75 not in alpha_grid:
            raise ValueError("D046 alpha_grid must retain the D043 alpha=0.75 anchor")
        object.__setattr__(self, "alpha_grid", alpha_grid)

        confidence = float(self.confidence_level)
        epsilon = float(self.relative_epsilon)
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("relative_epsilon must be finite and strictly positive")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "relative_epsilon", epsilon)

        if self.topology_labels != ("R", "SW", "SF"):
            raise ValueError("D046 requires the frozen R/SW/SF topology order")
        if self.topology_pairs != (("R", "SW"), ("R", "SF"), ("SW", "SF")):
            raise ValueError("D046 requires all three topology contrasts")
        if len(self.outcomes) == 0 or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be non-empty and unique")

    @property
    def n_alpha(self) -> int:
        return len(self.alpha_grid)

    @property
    def n_matched_blocks(self) -> int:
        return self.n_alpha * self.n_replications

    @property
    def n_simulations(self) -> int:
        return self.n_matched_blocks * len(self.topology_labels)

    def uses_relative_effect(self, outcome: str) -> bool:
        """Reuse the frozen D045 relative-effect convention."""

        if outcome not in self.outcomes:
            raise KeyError(outcome)
        return first_confirmatory_production_protocol().uses_relative_effect(outcome)


def first_alpha_sweep_protocol() -> AlphaSweepProtocol:
    """Return the frozen D046 exploratory alpha-sweep design."""

    return AlphaSweepProtocol()
