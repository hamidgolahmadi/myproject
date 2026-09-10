"""Frozen D049 reduced-factorial alpha x beta x gamma_R interaction protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .confirmatory_protocol import first_confirmatory_production_protocol
from .gamma_sweep_protocol import GAMMA_DIAGNOSTIC_OUTCOMES


JOINT_EXPERIMENT_SEED = 2026091003
JOINT_BOOTSTRAP_SEED = 2026091004
FACTORIAL_ALPHA_LEVELS = (0.40, 0.85, 0.99)
FACTORIAL_BETA_LEVELS = (0.0, 1.0, 5.0, 100.0)
FACTORIAL_GAMMA_LEVELS = (0.0, 0.90, 0.99, 0.999)
JOINT_CORE_OUTCOMES = (
    "mean_attention_overlap",
    "mean_pairwise_action_covariance",
    "mean_aggregate_order_flow_variance",
    "mean_absolute_order_flow_per_agent",
    "return_volatility",
    "peak_cid",
)


@dataclass(frozen=True, slots=True)
class JointCell:
    cell_id: str
    cell_kind: str
    alpha: float
    beta: float
    gamma_R: float

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise ValueError("cell_id must be a non-empty string")
        if self.cell_kind not in {"factorial", "alpha0_control", "d043_anchor"}:
            raise ValueError("invalid D049 cell_kind")
        alpha = float(self.alpha)
        beta = float(self.beta)
        gamma_R = float(self.gamma_R)
        if not np.isfinite(alpha) or not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if not np.isfinite(gamma_R) or not 0.0 <= gamma_R < 1.0:
            raise ValueError("gamma_R must satisfy 0 <= gamma_R < 1")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "gamma_R", gamma_R)


@dataclass(frozen=True, slots=True)
class JointLinearContrastSpec:
    """Frozen linear contrast of SF-SW cell effects."""

    name: str
    contrast_kind: str
    terms: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("contrast name must be non-empty")
        if self.contrast_kind not in {"interaction", "boundary_shift"}:
            raise ValueError("invalid contrast_kind")
        if len(self.terms) < 2:
            raise ValueError("a D049 contrast needs at least two terms")
        total = 0.0
        for alpha, beta, gamma_R, coefficient in self.terms:
            JointCell("validation", "factorial", alpha, beta, gamma_R)
            coefficient = float(coefficient)
            if not np.isfinite(coefficient) or coefficient == 0.0:
                raise ValueError("contrast coefficients must be finite and non-zero")
            total += coefficient
        if not np.isclose(total, 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("D049 interaction/shift coefficients must sum to zero")


def _factorial_cells() -> tuple[JointCell, ...]:
    cells: list[JointCell] = []
    for index, (alpha, beta, gamma_R) in enumerate(
        product(FACTORIAL_ALPHA_LEVELS, FACTORIAL_BETA_LEVELS, FACTORIAL_GAMMA_LEVELS)
    ):
        cells.append(
            JointCell(
                cell_id=f"F{index:02d}",
                cell_kind="factorial",
                alpha=alpha,
                beta=beta,
                gamma_R=gamma_R,
            )
        )
    return tuple(cells)


FROZEN_JOINT_CELLS = _factorial_cells() + (
    JointCell("C_ALPHA0", "alpha0_control", 0.0, 5.0, 0.90),
    JointCell("A_D043", "d043_anchor", 0.75, 1.0, 0.90),
)


def _did_terms(
    *,
    a0: float,
    a1: float,
    b0: float,
    b1: float,
    gamma: float,
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (a1, b1, gamma, +1.0),
        (a0, b1, gamma, -1.0),
        (a1, b0, gamma, -1.0),
        (a0, b0, gamma, +1.0),
    )


def _beta_gamma_terms(
    *,
    alpha: float,
    b0: float,
    b1: float,
    g0: float,
    g1: float,
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (alpha, b1, g1, +1.0),
        (alpha, b0, g1, -1.0),
        (alpha, b1, g0, -1.0),
        (alpha, b0, g0, +1.0),
    )


def _alpha_gamma_terms(
    *,
    a0: float,
    a1: float,
    beta: float,
    g0: float,
    g1: float,
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (a1, beta, g1, +1.0),
        (a0, beta, g1, -1.0),
        (a1, beta, g0, -1.0),
        (a0, beta, g0, +1.0),
    )


def _three_way_terms() -> tuple[tuple[float, float, float, float], ...]:
    terms: list[tuple[float, float, float, float]] = []
    for alpha, a_sign in ((0.40, -1.0), (0.85, +1.0)):
        for beta, b_sign in ((0.0, -1.0), (5.0, +1.0)):
            for gamma_R, g_sign in ((0.0, -1.0), (0.99, +1.0)):
                terms.append((alpha, beta, gamma_R, a_sign * b_sign * g_sign))
    return tuple(terms)


FROZEN_INTERACTION_SPECS = (
    JointLinearContrastSpec(
        "beta_gamma_core_at_alpha085",
        "interaction",
        _beta_gamma_terms(alpha=0.85, b0=0.0, b1=5.0, g0=0.0, g1=0.99),
    ),
    JointLinearContrastSpec(
        "alpha_beta_core_at_gamma09",
        "interaction",
        _did_terms(a0=0.40, a1=0.85, b0=0.0, b1=5.0, gamma=0.90),
    ),
    JointLinearContrastSpec(
        "alpha_gamma_core_at_beta5",
        "interaction",
        _alpha_gamma_terms(a0=0.40, a1=0.85, beta=5.0, g0=0.0, g1=0.99),
    ),
    JointLinearContrastSpec(
        "alpha_beta_gamma_core",
        "interaction",
        _three_way_terms(),
    ),
    JointLinearContrastSpec(
        "beta_gamma_boundary_at_alpha085",
        "interaction",
        _beta_gamma_terms(alpha=0.85, b0=5.0, b1=100.0, g0=0.99, g1=0.999),
    ),
    JointLinearContrastSpec(
        "alpha_gamma_boundary_at_beta5",
        "interaction",
        _alpha_gamma_terms(a0=0.85, a1=0.99, beta=5.0, g0=0.99, g1=0.999),
    ),
    JointLinearContrastSpec(
        "gamma_boundary_shift_at_alpha085_beta5",
        "boundary_shift",
        ((0.85, 5.0, 0.999, +1.0), (0.85, 5.0, 0.99, -1.0)),
    ),
    JointLinearContrastSpec(
        "alpha_boundary_shift_at_beta5_gamma09",
        "boundary_shift",
        ((0.99, 5.0, 0.90, +1.0), (0.85, 5.0, 0.90, -1.0)),
    ),
    JointLinearContrastSpec(
        "beta_saturation_shift_at_alpha085_gamma09",
        "boundary_shift",
        ((0.85, 100.0, 0.90, +1.0), (0.85, 5.0, 0.90, -1.0)),
    ),
)


def _d049_outcomes() -> tuple[str, ...]:
    return first_confirmatory_production_protocol().all_outcomes + GAMMA_DIAGNOSTIC_OUTCOMES


@dataclass(frozen=True, slots=True)
class JointInteractionProtocol:
    experiment_seed: int = JOINT_EXPERIMENT_SEED
    bootstrap_seed: int = JOINT_BOOTSTRAP_SEED
    cells: tuple[JointCell, ...] = FROZEN_JOINT_CELLS
    interaction_specs: tuple[JointLinearContrastSpec, ...] = FROZEN_INTERACTION_SPECS
    n_replications: int = 300
    n_bootstrap: int = 5_000
    confidence_level: float = 0.95
    relative_epsilon: float = 1e-12
    topology_labels: tuple[str, ...] = ("R", "SW", "SF")
    topology_pairs: tuple[tuple[str, str], ...] = (
        ("R", "SW"),
        ("R", "SF"),
        ("SW", "SF"),
    )
    interaction_topology_pair: tuple[str, str] = ("SF", "SW")
    core_outcomes: tuple[str, ...] = JOINT_CORE_OUTCOMES
    outcomes: tuple[str, ...] = field(default_factory=_d049_outcomes)

    def __post_init__(self) -> None:
        for name in ("experiment_seed", "bootstrap_seed", "n_replications", "n_bootstrap"):
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
            raise ValueError("experiment and bootstrap seeds must be disjoint")
        if self.topology_labels != ("R", "SW", "SF"):
            raise ValueError("D049 requires frozen R/SW/SF topology order")
        if self.topology_pairs != (("R", "SW"), ("R", "SF"), ("SW", "SF")):
            raise ValueError("D049 requires all three cellwise topology contrasts")
        if self.interaction_topology_pair != ("SF", "SW"):
            raise ValueError("D049 interaction estimands are frozen on SF-SW")
        if len(self.cells) != 50:
            raise ValueError("D049 requires exactly 50 parameter cells")
        if len({cell.cell_id for cell in self.cells}) != 50:
            raise ValueError("D049 cell ids must be unique")
        factorial = tuple(cell for cell in self.cells if cell.cell_kind == "factorial")
        if len(factorial) != 48:
            raise ValueError("D049 requires exactly 48 factorial cells")
        expected_factorial = {
            (a, b, g)
            for a in FACTORIAL_ALPHA_LEVELS
            for b in FACTORIAL_BETA_LEVELS
            for g in FACTORIAL_GAMMA_LEVELS
        }
        if {(c.alpha, c.beta, c.gamma_R) for c in factorial} != expected_factorial:
            raise ValueError("D049 factorial grid does not match the frozen reduced design")
        alpha0 = [cell for cell in self.cells if cell.cell_kind == "alpha0_control"]
        anchor = [cell for cell in self.cells if cell.cell_kind == "d043_anchor"]
        if len(alpha0) != 1 or (alpha0[0].alpha, alpha0[0].beta, alpha0[0].gamma_R) != (0.0, 5.0, 0.90):
            raise ValueError("D049 alpha-zero control is missing or altered")
        if len(anchor) != 1 or (anchor[0].alpha, anchor[0].beta, anchor[0].gamma_R) != (0.75, 1.0, 0.90):
            raise ValueError("D049 D043 anchor is missing or altered")
        factorial_lookup = {(c.alpha, c.beta, c.gamma_R) for c in factorial}
        for spec in self.interaction_specs:
            for alpha, beta, gamma_R, _ in spec.terms:
                if (alpha, beta, gamma_R) not in factorial_lookup:
                    raise ValueError(f"interaction {spec.name} references a non-factorial cell")
        if len({spec.name for spec in self.interaction_specs}) != len(self.interaction_specs):
            raise ValueError("D049 interaction names must be unique")
        confidence = float(self.confidence_level)
        epsilon = float(self.relative_epsilon)
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("relative_epsilon must be finite and positive")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "relative_epsilon", epsilon)
        if len(self.outcomes) == 0 or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes must be non-empty and unique")
        if not set(self.core_outcomes).issubset(self.outcomes):
            raise ValueError("all D049 core outcomes must be retained")
        for diagnostic in GAMMA_DIAGNOSTIC_OUTCOMES:
            if diagnostic not in self.outcomes:
                raise ValueError("D049 must retain D048 reputation-scale diagnostics")

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_checkpoints(self) -> int:
        return self.n_cells * self.n_replications

    @property
    def n_simulations(self) -> int:
        return self.n_checkpoints * len(self.topology_labels)

    def cell_index(self, cell_id: str) -> int:
        for index, cell in enumerate(self.cells):
            if cell.cell_id == cell_id:
                return index
        raise KeyError(cell_id)

    def factorial_cell(self, alpha: float, beta: float, gamma_R: float) -> JointCell:
        target = (float(alpha), float(beta), float(gamma_R))
        for cell in self.cells:
            if cell.cell_kind == "factorial" and (cell.alpha, cell.beta, cell.gamma_R) == target:
                return cell
        raise KeyError(target)

    def uses_relative_effect(self, outcome: str) -> bool:
        if outcome not in self.outcomes:
            raise KeyError(outcome)
        if outcome in GAMMA_DIAGNOSTIC_OUTCOMES:
            return True
        return first_confirmatory_production_protocol().uses_relative_effect(outcome)


def first_joint_interaction_protocol() -> JointInteractionProtocol:
    """Return the frozen D049 reduced-factorial interaction design."""

    return JointInteractionProtocol()
