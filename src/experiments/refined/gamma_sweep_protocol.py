"""Frozen D048 exploratory reputation-persistence sweep protocol.

D048 follows completed D046/D047 regime mapping. It fixes alpha at 0.85 and
beta at the post-D047 interior amplification anchor 5.0, then varies gamma_R
while every other D043 parameter and the D044 market-evaluation calibration
remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .confirmatory_protocol import first_confirmatory_production_protocol


GAMMA_SWEEP_EXPERIMENT_SEED = 2026091001
GAMMA_SWEEP_BOOTSTRAP_SEED = 2026091002
GAMMA_SWEEP_ALPHA_ANCHOR = 0.85
GAMMA_SWEEP_BETA_ANCHOR = 5.0
FROZEN_GAMMA_GRID = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
GAMMA_DIAGNOSTIC_OUTCOMES = (
    "mean_raw_local_reputation_std",
    "mean_raw_local_reputation_std_over_sigma0",
)


def _d048_outcomes() -> tuple[str, ...]:
    return first_confirmatory_production_protocol().all_outcomes + GAMMA_DIAGNOSTIC_OUTCOMES


@dataclass(frozen=True, slots=True)
class GammaSweepProtocol:
    """Exploratory complete-block OAT design for D048."""

    experiment_seed: int = GAMMA_SWEEP_EXPERIMENT_SEED
    alpha_anchor: float = GAMMA_SWEEP_ALPHA_ANCHOR
    beta_anchor: float = GAMMA_SWEEP_BETA_ANCHOR
    gamma_grid: tuple[float, ...] = FROZEN_GAMMA_GRID
    n_replications: int = 300
    bootstrap_seed: int = GAMMA_SWEEP_BOOTSTRAP_SEED
    n_bootstrap: int = 5_000
    confidence_level: float = 0.95
    relative_epsilon: float = 1e-12
    topology_labels: tuple[str, ...] = ("R", "SW", "SF")
    topology_pairs: tuple[tuple[str, str], ...] = (
        ("R", "SW"),
        ("R", "SF"),
        ("SW", "SF"),
    )
    outcomes: tuple[str, ...] = field(default_factory=_d048_outcomes)

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
        beta_anchor = float(self.beta_anchor)
        if not np.isfinite(alpha_anchor) or alpha_anchor != GAMMA_SWEEP_ALPHA_ANCHOR:
            raise ValueError("D048 requires the frozen post-D046 alpha anchor 0.85")
        if not np.isfinite(beta_anchor) or beta_anchor != GAMMA_SWEEP_BETA_ANCHOR:
            raise ValueError("D048 requires the post-D047 interior beta anchor 5.0")
        object.__setattr__(self, "alpha_anchor", alpha_anchor)
        object.__setattr__(self, "beta_anchor", beta_anchor)

        try:
            gamma_grid = tuple(float(value) for value in self.gamma_grid)
        except (TypeError, ValueError) as exc:
            raise TypeError("gamma_grid must contain real scalars") from exc
        if len(gamma_grid) < 2:
            raise ValueError("gamma_grid must contain at least two values")
        if any(not np.isfinite(value) or not 0.0 <= value < 1.0 for value in gamma_grid):
            raise ValueError("every gamma_R value must satisfy 0 <= gamma_R < 1")
        if len(set(gamma_grid)) != len(gamma_grid):
            raise ValueError("gamma_grid values must be unique")
        if tuple(sorted(gamma_grid)) != gamma_grid:
            raise ValueError("gamma_grid must be strictly increasing")
        if gamma_grid[0] != 0.0:
            raise ValueError("D048 gamma_grid must include gamma_R=0 as its no-memory control")
        if 0.9 not in gamma_grid:
            raise ValueError("D048 gamma_grid must retain the D043 gamma_R=0.9 baseline anchor")
        if gamma_grid[-1] != 0.999:
            raise ValueError("D048 gamma_grid must retain gamma_R=0.999 as the persistence stress point")
        object.__setattr__(self, "gamma_grid", gamma_grid)

        confidence = float(self.confidence_level)
        epsilon = float(self.relative_epsilon)
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("relative_epsilon must be finite and strictly positive")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "relative_epsilon", epsilon)

        if self.topology_labels != ("R", "SW", "SF"):
            raise ValueError("D048 requires the frozen R/SW/SF topology order")
        if self.topology_pairs != (("R", "SW"), ("R", "SF"), ("SW", "SF")):
            raise ValueError("D048 requires all three topology contrasts")
        if len(self.outcomes) == 0 or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be non-empty and unique")
        for diagnostic in GAMMA_DIAGNOSTIC_OUTCOMES:
            if diagnostic not in self.outcomes:
                raise ValueError("D048 outcomes must retain both reputation-scale diagnostics")

    @property
    def n_gamma(self) -> int:
        return len(self.gamma_grid)

    @property
    def n_matched_blocks(self) -> int:
        return self.n_gamma * self.n_replications

    @property
    def n_simulations(self) -> int:
        return self.n_matched_blocks * len(self.topology_labels)

    def uses_relative_effect(self, outcome: str) -> bool:
        """Use D045 conventions plus relative effects for positive D048 diagnostics."""

        if outcome not in self.outcomes:
            raise KeyError(outcome)
        if outcome in GAMMA_DIAGNOSTIC_OUTCOMES:
            return True
        return first_confirmatory_production_protocol().uses_relative_effect(outcome)


def first_gamma_sweep_protocol() -> GammaSweepProtocol:
    """Return the frozen D048 exploratory gamma_R-sweep design."""

    return GammaSweepProtocol()
