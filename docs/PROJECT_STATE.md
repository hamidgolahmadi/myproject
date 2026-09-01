# Project State

Last updated: 2026-09-01

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

Refined topology layer:

    src/topologies/refined/
        generators.py
        diagnostics.py

Structural-validation driver:

    scripts/run_refined_structural_validation.py

---

## 4. Binding Scientific Decisions

See `docs/DECISIONS.md`. Key frozen points include separation of `G` and `W_t`, lagged attention in beliefs, signed net order flow, inherited-position profit, paired common-random-number design, mandatory `alpha=0` negative control, and D041 structural-validation calibration.

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

Structural separation is strong and non-overlapping in the intended dimensions: every realised SF graph has greater in-degree Gini and top-five hub share than every realised Random graph, and every realised SW graph has greater clustering than every realised Random graph. All symmetrised supports are fully connected.

Interpret SW path length precisely: at `p_sw=0.02`, it is highly clustered and fully connected with a limited-shortcut structure, but APL-LCC is materially longer than Random. No recalibration is currently required merely to exaggerate separation.

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

Latest verified checkpoint:

    284 passed in 5.11s

with clean working tree and branch up to date with `origin/refined-model`.

---

## 8. Run-Level Market Outcome Metrics — VERIFIED

Implemented and VERIFIED in:

    src/experiments/refined/market_metrics.py

Implements:

    Eq. (236) return volatility RV
    Eq. (237) RMS mispricing RMSM
    Eq. (237) maximum absolute mispricing MAM
    Eq. (238) mean absolute signed net order flow per agent MAF
    Eq. (288) mean absolute return MAR
    Eq. (289) time-averaged cross-sectional belief variance V_b

Baseline evaluation uses `B=0`; positive burn-in remains an explicit robustness choice. Metrics evaluate `SimulationResult` only and do not duplicate economic dynamics.

---

## 9. Rolling Action Covariance — NEW

NEWLY IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION:

    src/experiments/refined/action_covariance.py

Public API:

    RollingActionCovariancePoint
    rolling_action_covariance(...)

Implements Section 5.5:

    Eq. (239) average pairwise sample action covariance over each full rolling window
    Eq. (240) exact sample variance decomposition of signed net order flow

For valid endpoints:

    t = B + L, ..., T

it uses the full window:

    {t-L+1, ..., t}

with sample covariance/variance denominator `L-1`.

The implementation checks that stored `F_t` equals the sum of agent actions and verifies numerically for every window that:

    Var_hat(F)
      =
    sum_i Var_hat(a_i)
      +
    N(N-1) C_a

It does not reconstruct market dynamics or substitute gross volume for signed order flow.

New test file:

    tests/test_refined_action_covariance.py

adds 18 pytest cases.

Expected next total:

    302 passed

---

## 10. Computational Milestones

Milestones 1-11 are VERIFIED, including structural validation and principal run-level market outcomes.

Milestone 12 — rolling action covariance and exact order-flow variance decomposition:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

Do not start the large refined market Monte Carlo yet. CID, influence/overlap/mobility diagnostics, and explicit market-run calibration remain to be fixed and verified.

---

## 11. Immediate Next Step

1. Pull latest `refined-model` on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       302 passed

4. If all 302 pass, record Milestone 12 as VERIFIED.
5. Then implement rolling CID components and normalisation, Eqs. (241)-(246), without choosing reference scales or thresholds from observed topology rankings.
6. After CID components, implement threshold/duration/censored-stabilisation logic, Eqs. (247)-(250).
7. Then implement effective-influence concentration, overlap, and mobility, Eqs. (251)-(265).
8. Freeze all window lengths, CID reference scales, CID weights, thresholds, guardrails, and stabilisation length before the topology-evaluation market Monte Carlo.

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
