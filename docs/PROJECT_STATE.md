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
        influence_metrics.py
        market_calibration.py

Refined topology layer:

    src/topologies/refined/
        generators.py
        diagnostics.py

Structural-validation driver:

    scripts/run_refined_structural_validation.py

---

## 4. Binding Scientific Decisions

See `docs/DECISIONS.md`.

Key frozen decisions include:

- G and W_t are distinct objects;
- beliefs use lagged W_{t-1};
- signed net order flow is F_t=sum_i a_i,t;
- profit uses inherited positions;
- paired common-random-number topology design;
- mandatory alpha=0 negative control;
- D041 first structural-validation calibration;
- D042 first refined market-evaluation calibration METHOD.

D042 freezes the calibration protocol, but the numerical CID reference scales and threshold do not exist yet. Actual market calibration must wait until the baseline refined parameter vector and non-network initial-condition rule are frozen.

---

## 5. Fixed-Topology Runtime — VERIFIED

Equations (35)-(82) are implemented and VERIFIED, including canonical timing, deterministic multi-period simulation, shock generation, paired semantic seeds, topology-specific graph generation, neutral graph-supported W_0, and generated-treatment alpha=0 controls.

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

Mean diagnostics from the completed 3000-graph Iridis run:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

Structural separation is strong and non-overlapping in the intended dimensions. All symmetrised supports are fully connected. SW is highly clustered and fully connected with a limited-shortcut structure, but its APL-LCC is materially longer than Random.

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
    284 passed   + run-level market outcomes
    302 passed   + rolling action covariance / order-flow variance decomposition
    341 passed   + rolling CID components / standardisation / CID
    374 passed   + threshold / duration / right-censored stabilisation
    409 passed   + realised-influence concentration / overlap / mobility

Latest verified checkpoint:

    409 passed in 6.04s

with clean working tree and branch up to date with `origin/refined-model`.

---

## 8. Evaluation and Mechanism Layers — VERIFIED

Verified modules:

    src/experiments/refined/market_metrics.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/cid.py
    src/experiments/refined/cid_events.py
    src/experiments/refined/influence_metrics.py

Implemented and VERIFIED:

- Eqs. (236)-(238): RV, RMSM, MAM, MAF;
- Eqs. (239)-(240): rolling action covariance and exact order-flow variance decomposition;
- Eqs. (241)-(246): raw CID components, reference-scale standardisation, weights, CID;
- Eqs. (247)-(250): threshold exceedance, peak/duration, operational stabilisation, right-censoring;
- Eqs. (251)-(265): normalised attention entropy, effective sources, realised influence shares/HHI, structural-hub realised influence, overlap, mobility;
- Eqs. (288)-(289): MAR and time-averaged cross-sectional belief variance.

Eq. (266)-(267) KL divergence to a transition prior remains deferred with the attention-inertia extension because the current first-stage rule is frictionless tau=0.

The mechanism chain can now be measured directly:

    G
      -> W_t concentration / structural-hub influence / overlap / mobility
      -> rolling action covariance
      -> signed aggregate order flow
      -> return / mispricing outcomes

---

## 9. Market-Evaluation Calibration Protocol — NEW

NEWLY IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION:

    src/experiments/refined/market_calibration.py

Decision register:

    D042 — First Refined Market-Evaluation Calibration Protocol

Public API:

    MarketEvaluationCalibrationProtocol
    MarketEvaluationCalibration
    first_market_evaluation_calibration_protocol(...)
    estimate_reference_scales(...)
    estimate_cid_threshold(...)
    calibrate_market_evaluation(...)

Frozen METHOD for the first confirmatory market run:

    T = 1000
    B = 0
    rolling L = 50
    robustness L = {25, 100}
    alpha_calibration = 0

    scale sample:
        500 no-social replications
        seed namespace 2026090201

    threshold sample:
        500 separate no-social replications
        seed namespace 2026090202

    reference scales:
        pooled median of each raw rolling component

    CID weights:
        equal (1/3, 1/3, 1/3)

    c_CID:
        95th percentile of run-level peak CID
        using the conservative `higher` finite-sample quantile convention

    baseline component guardrails:
        inactive

    L_stab = 50

The scale and threshold samples are deliberately separate. Their seeds must also remain disjoint from later topology-evaluation seeds.

A non-positive pooled-median reference scale is a calibration failure. The implementation does not add an epsilon to manufacture a valid denominator.

New test file:

    tests/test_refined_market_calibration.py

adds 36 pytest cases.

Expected next total:

    445 passed

---

## 10. Important Remaining Scientific Gate

DO NOT RUN the 500+500 no-social calibration simulations yet.

The calibration distribution depends on the maintained market specification. Before actual calibration, freeze explicitly:

1. the homogeneous baseline `RefinedParameters` vector:

       rho_theta
       sigma_theta
       v_bar
       psi
       sigma_s
       sigma_b
       alpha for evaluation regimes
       kappa
       x_bar
       chi
       lambda_price
       sigma_p
       gamma_R
       beta
       sigma_0

2. the non-network initial-condition rule for:

       theta_0
       b_0
       x_0
       p_0
       R_0

The report allows neutral x_0=0 and R_0=0 and stationary theta_0, but it does not uniquely specify numerical b_0 or p_0 rules. Do not silently invent them.

Legacy pilot defaults may inform provenance, but the refined price law, trading law, inventory constraint, profit convention, and attention rule differ. Therefore legacy numerical defaults must not be copied automatically into the refined baseline.

---

## 11. Computational Milestones

Milestones 1-15 are VERIFIED.

Milestone 15:

    realised-influence concentration, overlap, and mobility, Eqs. (251)-(265)
    VERIFIED at 409 tests

Milestone 16:

    separate-sample market-evaluation calibration protocol
    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

Large topology-evaluation market Monte Carlo remains prohibited until the market parameter vector, non-network initial rule, calibration outputs, and a small end-to-end market-run smoke test are all frozen and verified.

---

## 12. Immediate Next Step

1. Pull latest `refined-model` on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       445 passed

4. If 445 pass, record Milestone 16 as VERIFIED.
5. Then inspect the report and legacy pilot only for provenance and define the FIRST REFINED BASELINE market parameter vector and non-network initial-condition rule as a new explicit decision.
6. Only after that decision is tested should we build the no-social calibration runner for D042.
7. Run a small calibration smoke job before the full 500+500 calibration samples.
8. Freeze the resulting c_ret, c_bel, c_F, and c_CID in a persistent calibration artifact before any R/SW/SF market-evaluation job.

---

## 13. Planned Development Sequence

    Phase 1  Refined fixed-topology core model                         COMPLETE
    Phase 2  Refined binary topology generators                       COMPLETE
    Phase 3  Deterministic integration and multi-period tests         COMPLETE
    Phase 4  Paired fixed-topology design + structural validation      COMPLETE
    Phase 5  Refined market metrics / CID / calibration protocol       NEAR COMPLETE
    Phase 6  Influence / overlap / action-covariance diagnostics       COMPLETE
    Phase 7  Baseline market calibration + paired market runner        NEXT
    Phase 8  alpha / beta / gamma_R experiments and heterogeneity      PLANNED
    Phase 9  Endogenous feasible-network formation and rewiring        PLANNED
    Phase 10 Full equilibrium Jacobian / Lyapunov analysis             PLANNED
    Phase 11 State-space / synthetic recovery / EKF / empirical work   PLANNED
    Phase 12 Planner / policy analysis                                 PLANNED

Optional later extension: MARL

---

## 14. New-Chat Handoff Prompt

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
