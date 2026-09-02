# Project State

Last updated: 2026-09-02

## Project identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Branch:

    refined-model

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility only.

Iridis setup:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model

## Verified architecture

Core refined model Eqs. (35)-(82): VERIFIED.

Paired semantic seeds, common shock paths, topology-specific G/W0, generated-treatment alpha=0 control: VERIFIED.

Structural diagnostics and D041 3000-graph validation: VERIFIED.

Structural-validation means:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

Evaluation/mechanism layers VERIFIED:

- Eqs. (236)-(238), (288)-(289): run-level outcomes;
- Eqs. (239)-(240): rolling action covariance and exact net-flow variance decomposition;
- Eqs. (241)-(246): CID components, scales, weights, CID;
- Eqs. (247)-(250): exceedance, duration, stabilisation, right-censoring;
- Eqs. (251)-(265): attention entropy/effective sources, realised influence shares/HHI, structural-hub realised influence, overlap, mobility.

Eq. (266)-(267) KL-to-transition-prior remains deferred with attention inertia.

## Verified test checkpoints

Latest Iridis checkpoint:

    445 passed in 7.61s

with clean working tree and branch up to date with origin/refined-model.

This verifies Milestone 16: D042 separate-sample market-evaluation calibration METHOD.

D042 method:

    T = 1000
    B = 0
    rolling L = 50
    robustness L = {25, 100}
    alpha_calibration = 0
    scale sample = 500 runs, namespace 2026090201
    threshold sample = 500 separate runs, namespace 2026090202
    reference scales = pooled component medians
    CID weights = equal thirds
    c_CID = 95th percentile of run-level peak CID, quantile method higher
    component guardrails = inactive baseline
    L_stab = 50

DO NOT run the 500+500 calibration yet.

## Provisional refined baseline candidate — NEW

New module:

    src/experiments/refined/baseline_specification.py

Provenance note:

    docs/REFINED_BASELINE_CANDIDATE.md

Status:

    PROVISIONAL — NOT FROZEN

Candidate design:

    N = 100
    K = 6
    T = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

Candidate RefinedParameters:

    rho_theta    = 0.985
    sigma_theta  = 0.025
    v_bar        = 0.0
    psi          = 1.0
    sigma_s      = 0.06
    sigma_b      = 0.025
    alpha        = 0.75
    kappa        = 2.4
    x_bar        = 5.0
    chi          = 0.02
    lambda_price = 0.0002
    sigma_p      = 0.001
    gamma_R      = 0.9
    beta         = 1.0
    sigma_0      = 1e-6

Provisional neutral non-network initialisation:

    theta_0 ~ stationary AR(1)
    b_i,0 = theta_0 for all i
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

W_0 remains topology-specific uniform graph-supported attention.

Rationale: B=0 means arbitrary initial mispricing or disagreement would mechanically contaminate the measured path. This neutral rule starts without such artificial deviations; period-1 signals and belief noise then operate normally.

The pilot is used only as provenance. In particular, pilot level-price coefficients are mapped to refined normalised return units before being used as candidate anchors. New refined choices chi, x_bar, and sigma_0 must pass scale validation.

New tests:

    tests/test_refined_baseline_specification.py

adds 29 pytest cases.

Expected next total:

    474 passed

## Immediate next step

1. Pull latest refined-model on Iridis.
2. Run all refined tests.
3. Expected: 474 passed.
4. If green, build a SMALL paired scale/non-degeneracy smoke runner using this provisional candidate.
5. Smoke must report return/mispricing scales, signed net flow per agent, desired-action magnitudes/saturation, inventory-bound contacts, reputation scale relative to sigma_0, influence HHI/overlap/mobility, and finiteness.
6. Smoke is NOT a topology-ranking search. Do not tune the candidate to make R/SW/SF differ.
7. Only after the candidate passes scale smoke should it be promoted to a frozen design decision and used for D042 calibration.
8. Then run a small no-social calibration smoke before the full 500+500 calibration samples.

Large topology-evaluation Monte Carlo remains prohibited until these gates pass.

## Development status

    Phase 1  Refined fixed-topology core                         COMPLETE
    Phase 2  Topology generators                                COMPLETE
    Phase 3  Deterministic integration                          COMPLETE
    Phase 4  Paired design + structural validation              COMPLETE
    Phase 5  Market metrics / CID / calibration method          COMPLETE
    Phase 6  Mechanism diagnostics                              COMPLETE
    Phase 7  Baseline scale smoke + market calibration          IN PROGRESS
    Phase 8  Paired confirmatory market runner                  PLANNED
    Phase 9  alpha/beta/gamma experiments + heterogeneity       PLANNED
    Phase 10 Endogenous G formation                             PLANNED
    Phase 11 Full Jacobian / Lyapunov                           PLANNED
    Phase 12 State-space / EKF / empirical work                PLANNED
    Phase 13 Planner / policy                                   PLANNED
