"""Frozen D047 exploratory reputational-selectivity sweep protocol.

D047 follows the completed D046 alpha sweep.  It fixes alpha at the D046
post-analysis interior anchor 0.85 and maps topology differentiation over the
homogeneous reputation-selectivity parameter beta while every other D043
parameter and the D044 market-evaluation calibration remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .confirmatory_protocol import first_confirmatory_production_protocol


BETA_SWEEP_EXPERIMENT_SEED = 2026090701
BETA_SWEEP_BOOTSTRAP_SEED = 2026090702
BETA_SWEEP_ALPHA_ANCHOR = 0.85
FROZEN_BETA_GRID = (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0)


def _d045_outcomes() -> tuple[str, ...]:
    return first_confirmatory_production_protocol().all_outcomes


@dataclass(frozen=True, slots=True)
class BetaSweepProtocol:
    """Exploratory matched-block design for D047."""

    experiment_seed: int = BETA_SWEEP_EXPERIMENT_SEED
    alpha_anchor: float = BETA_SWEEP_ALPHA_ANCHOR
    beta_grid: tuple[float, ...] = FROZEN_BETA_GRID
    n_replications: int = 300
    bootstrap_seed: int = BETA_SWEEP_BOOTSTRAP_SEED
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

        alpha_anchor = float(self.alpha_anchor)
        if not np.isfinite(alpha_anchor) or not 0.0 <= alpha_anchor < 1.0:
            raise ValueError("alpha_anchor must satisfy 0 <= alpha < 1")
        if alpha_anchor != BETA_SWEEP_ALPHA_ANCHOR:
            raise ValueError("D047 requires the frozen D046-selected alpha anchor 0.85")
        object.__setattr__(self, "alpha_anchor", alpha_anchor)

        try:
            beta_grid = tuple(float(value) for value in self.beta_grid)
        except (TypeError, ValueError) as exc:
            raise TypeError("beta_grid must contain real scalars") from exc
        if len(beta_grid) < 2:
            raise ValueError("beta_grid must contain at least two values")
        if any(not np.isfinite(value) or value < 0.0 for value in beta_grid):
            raise ValueError("every beta value must be finite and non-negative")
        if len(set(beta_grid)) != len(beta_grid):
            raise ValueError("beta_grid values must be unique")
        if tuple(sorted(beta_grid)) != beta_grid:
            raise ValueError("beta_grid must be strictly increasing")
        if beta_grid[0] != 0.0:
            raise ValueError("D047 beta_grid must include beta=0 as its first control point")
        if 1.0 not in beta_grid:
            raise ValueError("D047 beta_grid must retain the D043 beta=1 baseline anchor")
        if 2.0 not in beta_grid or 5.0 not in beta_grid:
            raise ValueError("D047 beta_grid must resolve the report's beta=2 to 5 transition region")
        if beta_grid[-1] != 1000.0:
            raise ValueError("D047 beta_grid must retain the report-scale beta=1000 upper endpoint")
        object.__setattr__(self, "beta_grid", beta_grid)

        confidence = float(self.confidence_level)
        epsilon = float(self.relative_epsilon)
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("relative_epsilon must be finite and strictly positive")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "relative_epsilon", epsilon)

        if self.topology_labels != ("R", "SW", "SF"):
            raise ValueError("D047 requires the frozen R/SW/SF topology order")
        if self.topology_pairs != (("R", "SW"), ("R", "SF"), ("SW", "SF")):
            raise ValueError("D047 requires all three topology contrasts")
        if len(self.outcomes) == 0 or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be non-empty and unique")

    @property
    def n_beta(self) -> int:
        return len(self.beta_grid)

    @property
    def n_matched_blocks(self) -> int:
        return self.n_beta * self.n_replications

    @property
    def n_simulations(self) -> int:
        return self.n_matched_blocks * len(self.topology_labels)

    def uses_relative_effect(self, outcome: str) -> bool:
        """Reuse the frozen D045 relative-effect convention."""

        if outcome not in self.outcomes:
            raise KeyError(outcome)
        return first_confirmatory_production_protocol().uses_relative_effect(outcome)


def first_beta_sweep_protocol() -> BetaSweepProtocol:
    """Return the frozen D047 exploratory beta-sweep design."""

    return BetaSweepProtocol()
