"""Frozen D045 first confirmatory fixed-topology production protocol.

The doctoral report fixes the paired estimand, triplet-preserving bootstrap,
three pairwise topology contrasts, reporting requirements, and the need for
predeclared multiplicity control.  It does not prescribe one numerical Monte
Carlo sample size or bootstrap count.  D045 therefore freezes those remaining
design choices before any baseline production outcomes are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONFIRMATORY_SMOKE_SEED = 2026090401
CONFIRMATORY_PRODUCTION_SEED = 2026090402
CONFIRMATORY_BOOTSTRAP_SEED = 2026090403


@dataclass(frozen=True, slots=True)
class ConfirmatoryProductionProtocol:
    """Predeclared production and inference design for the first benchmark run."""

    experiment_seed: int = CONFIRMATORY_PRODUCTION_SEED
    n_replications: int = 1000
    bootstrap_seed: int = CONFIRMATORY_BOOTSTRAP_SEED
    n_bootstrap: int = 10_000
    confidence_level: float = 0.95
    familywise_alpha: float = 0.05
    relative_epsilon: float = 1e-12

    topology_labels: tuple[str, ...] = ("R", "SW", "SF")
    topology_pairs: tuple[tuple[str, str], ...] = (
        ("R", "SW"),
        ("R", "SF"),
        ("SW", "SF"),
    )

    primary_outcomes: tuple[str, ...] = (
        "return_volatility",
        "rms_mispricing",
        "maximum_absolute_mispricing",
        "mean_absolute_order_flow_per_agent",
        "peak_cid",
        "threshold_exceeding",
    )

    mechanism_outcomes: tuple[str, ...] = (
        "mean_hub_influence_share",
        "mean_attention_overlap",
        "mean_pairwise_action_covariance",
        "mean_aggregate_order_flow_variance",
    )

    secondary_outcomes: tuple[str, ...] = (
        "mean_absolute_return",
        "time_averaged_belief_variance",
        "cid_exceedance_duration_share",
        "right_censored",
        "mean_influence_hhi",
        "mean_attention_entropy",
        "mean_effective_sources",
        "mean_attention_mobility",
        "mean_sum_individual_action_variances",
    )

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
            raise ValueError("n_bootstrap must be at least 1000 for the frozen percentile design")
        if self.experiment_seed in {CONFIRMATORY_SMOKE_SEED, self.bootstrap_seed}:
            raise ValueError("production, smoke, and bootstrap seed namespaces must be disjoint")
        if self.bootstrap_seed == CONFIRMATORY_SMOKE_SEED:
            raise ValueError("bootstrap and smoke seed namespaces must be disjoint")

        confidence = float(self.confidence_level)
        familywise_alpha = float(self.familywise_alpha)
        epsilon = float(self.relative_epsilon)
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not np.isfinite(familywise_alpha) or not 0.0 < familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie strictly between zero and one")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("relative_epsilon must be finite and strictly positive")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "familywise_alpha", familywise_alpha)
        object.__setattr__(self, "relative_epsilon", epsilon)

        if self.topology_labels != ("R", "SW", "SF"):
            raise ValueError("D045 requires the frozen R/SW/SF topology order")
        if self.topology_pairs != (("R", "SW"), ("R", "SF"), ("SW", "SF")):
            raise ValueError("D045 requires all three predeclared pairwise topology contrasts")

        outcome_groups = (
            self.primary_outcomes,
            self.mechanism_outcomes,
            self.secondary_outcomes,
        )
        for group in outcome_groups:
            if len(group) == 0 or len(set(group)) != len(group):
                raise ValueError("outcome groups must be non-empty and internally unique")
        all_outcomes = sum(outcome_groups, ())
        if len(set(all_outcomes)) != len(all_outcomes):
            raise ValueError("primary, mechanism, and secondary outcome groups must be disjoint")

    @property
    def primary_family_size(self) -> int:
        return len(self.primary_outcomes) * len(self.topology_pairs)

    @property
    def mechanism_family_size(self) -> int:
        return len(self.mechanism_outcomes) * len(self.topology_pairs)


def first_confirmatory_production_protocol() -> ConfirmatoryProductionProtocol:
    """Return the frozen D045 first confirmatory production design."""

    return ConfirmatoryProductionProtocol()
