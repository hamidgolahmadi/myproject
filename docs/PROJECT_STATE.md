# Project State

Last updated: 2026-09-02

## 1. Project Identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Primary branch:

    refined-model

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility only and must not override the report.

---

## 2. Iridis Session Setup

At the beginning of an Iridis session:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model
    git status

`unset PYTHONPATH` remains mandatory because the Iridis Python module injects a system PYTHONPATH.

---

## 3. Refined Architecture

Core model:

    src/model/refined/

Refined experiment/evaluation layer:

    src/experiments/refined/
        seeding.py
        paired.py
        treatments.py
        structural.py
        calibration.py
        structural_io.py
        market_metrics.py
        action_covariance.py
        cid.py
        cid_events.py

Refined topology layer:

    src/topologies/refined/
        generators.py
        diagnostics.py

Structural-validation driver:

    scripts/run_refined_structural_validation.py

---

## 4. Binding Scientific Decisions

See `docs/DECISIONS.md`. Key frozen points include separation of `G` and `W_t`, lagged attention in beliefs, signed net order flow, inherited-position profit, paired common-random-number design, mandatory `alpha=0` negative control, and D041 structural-validation calibration.

No numerical CID threshold, component guardrail, or reference scale has yet been frozen for the market experiment.

---

## 5. Verified Refined Core and Paired Infrastructure

The fixed-topology runtime for Equations (35)-(82) is implemented and VERIFIED, including canonical one-period timing, deterministic multi-period simulation, shock generation, paired semantic seeds, topology-specific graph generation, neutral graph-supported `W_0`, and generated-treatment `alpha=0` controls.

---

## 6. Structural Validation — COMPLETED SUCCESSFULLY

D041:

    experiment_seed = 20260901
    N = 100
    K = 6
    n_graph_replications = 1000 per topology
    q = 5
    p_sw = 0.02
    a0 = 1.0

The 3000-graph structural run completed on Iridis.

Mean diagnostics:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

Structural separation is strong and non-overlapping in the intended dimensions. All symmetrised supports are fully connected. SW is highly clustered and fully connected with a limited-shortcut structure, but its APL-LCC is materially longer than Random; do not describe it as Random-like in path length.

---

## 7. Verified Test Checkpoints

Verified on Iridis:

    21 passed
    30 passed
    40 passed
    51 passed
    66 passed
    82 passed
    90 passed
    100 passed
    115 passed
    138 passed
    166 passed
    199 passed
    219 passed
    247 passed
    261 passed
    284 passed   + run-level market outcomes, Eqs. (236)-(238), (288)-(289)
    302 passed   + rolling action covariance / order-flow variance decomposition, Eqs. (239)-(240)
    341 passed   + rolling CID components, standardisation, and dimensionless CID, Eqs. (241)-(246)

Latest verified checkpoint:

    341 passed in 6.59s

with clean working tree and branch up to date with `origin/refined-model`.

---

## 8. Run-Level and Rolling Market Metrics — VERIFIED

Verified modules:

    src/experiments/refined/market_metrics.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/cid.py

Implemented and VERIFIED:

    Eq. (236) return volatility RV
    Eq. (237) RMS mispricing RMSM
    Eq. (237) maximum absolute mispricing MAM
    Eq. (238) mean absolute signed net order flow per agent MAF
    Eq. (239) rolling average pairwise sample action covariance
    Eq. (240) exact rolling signed-order-flow variance decomposition
    Eq. (241) rolling sample return volatility V_ret
    Eq. (242) rolling belief dispersion B_bel
    Eq. (243) RMS signed net-order-flow pressure Q_F
    Eq. (244) explicit positive reference scales
    Eq. (245) non-negative weights summing to one
    Eq. (246) dimensionless weighted CID
    Eq. (288) mean absolute return MAR
    Eq. (289) time-averaged cross-sectional belief variance V_b

Eq. (242) uses period-level population cross-sectional belief variance `(1/N) sum_i (b_i-bbar)^2`, consistent with Eq. (289).

No CID reference scales are hard-coded. `CIDWeights.equal()` is only a convenience constructor and does not itself freeze the experiment design.

---

## 9. Threshold Exceedance and Operational Stabilisation — NEW

NEWLY IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION:

    src/experiments/refined/cid_events.py

Public API:

    CIDThresholdConfiguration
    OperationalStabilisationResult
    CIDRunClassification
    operational_stabilisation(...)
    classify_cid_path(...)
    threshold_exceedance_rate(...)

Implements:

    Eq. (247) run-level threshold-exceedance indicator using OR across CID and active component guardrails
    Eq. (248) topology-level exceedance rate as the mean of run indicators
    Eq. (249) peak CID and fraction of evaluated windows with CID > c_CID
    Eq. (250) first operational stabilisation start requiring CID and all active guardrails to remain admissible for L_stab consecutive windows

Important semantics:

- threshold exceedance uses strict `>`;
- stabilisation admissibility uses `<=`;
- Eq. (249) duration share counts CID threshold crossings only, not component-guardrail crossings;
- inactive component guardrails are represented as `None` and treated as `+infinity`;
- if no qualifying stabilisation block exists, `stabilisation_period=None` and `right_censored=True`; no artificial zero or pseudo-time is created;
- `last_eligible_start_period` records the final start for which a complete L_stab block could be observed;
- first-stage `L_stab=50` is the report-defined default, not an inferred calibration.

No numerical `c_CID` or component guardrail values are hard-coded.

New test file:

    tests/test_refined_cid_events.py

adds 33 pytest cases.

Expected next total:

    374 passed

---

## 10. Computational Milestones

Milestones 1-13 are VERIFIED, including structural validation, principal market outcomes, action covariance, and CID Eqs. (241)-(246).

Milestone 14 — threshold/duration/right-censored stabilisation logic, Eqs. (247)-(250):

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

Do not start the large refined market Monte Carlo yet. Realised-influence diagnostics and explicit market-run calibration still need implementation/verification.

---

## 11. Immediate Next Step

1. Pull latest `refined-model` on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       374 passed

4. If all 374 pass, record Milestone 14 as VERIFIED.
5. Then implement effective-influence concentration, overlap, and mobility, Eqs. (251)-(265).
6. After mechanism diagnostics, freeze the market-evaluation calibration inputs before topology-evaluation Monte Carlo: `L`, CID reference scales, CID weights, `c_CID`, optional component guardrails, and the report-defined `L_stab=50`.
7. Calibration must be independent of observed topology rankings.

---

## 12. Planned Development Sequence

    Phase 1  Refined fixed-topology core model                         COMPLETE
    Phase 2  Refined binary topology generators                       COMPLETE
    Phase 3  Deterministic integration and multi-period tests         COMPLETE
    Phase 4  Paired fixed-topology design + structural validation      COMPLETE
    Phase 5  Refined market metrics and CID                            IN PROGRESS
    Phase 6  Influence / overlap / action-covariance diagnostics       IN PROGRESS
    Phase 7  alpha / beta / gamma_R experiments and heterogeneity      PLANNED
    Phase 8  Endogenous feasible-network formation and rewiring        PLANNED
    Phase 9  Full equilibrium Jacobian / Lyapunov analysis             PLANNED
    Phase 10 State-space / synthetic recovery / EKF / empirical work   PLANNED
    Phase 11 Planner / policy analysis                                 PLANNED

Optional later extension: MARL

---

## 13. New-Chat Handoff Prompt

When starting a new conversation:

    We are implementing my 141-page PhD report on Iridis.

    Project root:
    /iridisfs/home/hg2e25/projects/myproject

    Current branch:
    refined-model

    The doctoral report is the scientific source of truth.
    Legacy code is reference only and must not override the report.

    Read and follow:
    docs/PROJECT_STATE.md
    docs/IMPLEMENTATION_MAP.md
    docs/DECISIONS.md

    Continue from the NEXT STEP in PROJECT_STATE.md.
