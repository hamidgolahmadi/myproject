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

## Verified test checkpoint

Latest Iridis checkpoint:

    474 passed in 8.02s

with clean working tree and branch up to date with origin/refined-model.

This verifies:

- Milestone 16: D042 separate-sample market-evaluation calibration METHOD;
- the provisional baseline-specification object and neutral non-network initialisation rule.

## D042 market-evaluation calibration method — FROZEN

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

## Provisional refined baseline candidate

Module:

    src/experiments/refined/baseline_specification.py

Provenance:

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

Candidate parameters:

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

Neutral non-network initialisation candidate:

    theta_0 ~ stationary AR(1)
    b_i,0 = theta_0 for all i
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

W_0 remains topology-specific uniform graph-supported attention.

## Baseline scale/non-degeneracy smoke — NEW

New module:

    src/experiments/refined/market_smoke.py

Driver:

    scripts/run_refined_baseline_scale_smoke.py

Tests:

    tests/test_refined_market_smoke.py

The default smoke design uses:

    experiment_seed = 2026090203
    paired replications = 5
    topology treatments = R, SW, SF
    candidate N = 100
    candidate T = 1000

The smoke seed namespace is separate from both D042 calibration namespaces and must not be reused for confirmatory topology evaluation.

The smoke records absolute scale diagnostics only. It does NOT estimate pairwise topology contrasts or rank topologies. Raw records retain topology labels only for anomaly tracing; the main summary is pooled across all smoke runs.

Diagnostics include:

- return sample SD, mean absolute return, maximum absolute return;
- RMS and maximum absolute mispricing;
- mean absolute and RMS signed net flow per agent;
- 95th percentile of absolute desired action;
- desired-action near-saturation fraction, using |a_tilde| >= 0.99 as an engineering diagnostic;
- execution-projection fraction, using executed action != desired action;
- fraction of realised positions on the inventory boundary;
- median and maximum local raw reputation dispersion;
- median local reputation scale relative to sigma_0;
- mean/max attention mobility and final W distance from W_0.

Only mathematical non-degeneracy is enforced in code: finite outputs, nonzero return variation, and nonzero order-flow variation. No arbitrary economic acceptance band is hard-coded.

New smoke tests add 30 pytest cases.

Expected next checkpoint:

    504 passed

## Immediate next step

1. Pull latest refined-model on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected: 504 passed.
4. If green, run the actual small smoke:

       python scripts/run_refined_baseline_scale_smoke.py

5. Inspect the pooled absolute-scale summary and raw records only for degeneracy/scale problems. Do not choose parameters because one topology ranks above another.
6. If scale is defensible, promote the candidate baseline to a frozen decision.
7. Then build/run a small no-social D042 calibration smoke before the full 500+500 calibration.
8. Persist and freeze numerical c_ret, c_bel, c_F, and c_CID before any confirmatory R/SW/SF market experiment.

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
