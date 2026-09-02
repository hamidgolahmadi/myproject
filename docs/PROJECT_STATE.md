# Project State

Last updated: 2026-09-02

## Project identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Branch:

    refined-model

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility only and must never override the report.

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

    530 passed in 8.32s

with clean working tree and branch up to date with `origin/refined-model`.

This verifies D043 implementation as well as all earlier model, evaluation, mechanism, smoke, and calibration-method layers.

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

The numerical `c_ret`, `c_bel`, `c_F`, and `c_CID` do not exist yet.

## D043 first refined baseline — FROZEN AND VERIFIED

Canonical module:

    src/experiments/refined/baseline_specification.py

Canonical documentation:

    docs/REFINED_BASELINE.md

Canonical API:

    RefinedBaselineSpecification
    first_refined_baseline_specification()

Frozen design:

    N = 100
    K = 6
    T = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

Frozen RefinedParameters:

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
    sigma_0      = 0.0005

Frozen neutral non-network initialisation:

    theta_0 ~ stationary AR(1)
    b_i,0 = theta_0 for all i
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

`W_0` remains topology-specific uniform graph-supported attention.

## Evidence for sigma_0 freeze

The common-random-number OAT sensitivity compared:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

using the same graph realisations, shocks, initial states, and all other parameters.

Market-scale outcomes were essentially invariant while attention regularisation changed smoothly. At `sigma_0=5e-4`, pooled medians were approximately:

    raw local reputation std / sigma_0    1.357
    mean attention mobility                0.02950
    max attention mobility                 0.32234
    final W distance from W0               0.29332
    return SD                              0.003428
    RMS mispricing                         0.068518
    projection fraction                    0.14385

The value is frozen because the floor is of the same order as realised local reputation dispersion and therefore regularises near-degenerate score normalisation without dominating adaptive attention.

## Report revision TODO — sigma_0 appendix

When the doctoral report is next revised, add the complete pre-freeze `sigma_0` sensitivity table to an Appendix / implementation-robustness section.

Required grid:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

Report at minimum:

    median local reputation dispersion / sigma_0
    mean attention mobility
    max attention mobility
    final W distance from W_0
    return SD
    RMS mispricing
    RMS signed net flow per agent
    desired-action p95
    execution-projection fraction
    inventory-boundary fraction

Use experiment seed `2026090203` and the completed 5-paired-replication CRN sensitivity run. The table is a regularisation-sensitivity diagnostic, not a topology-ranking table.

## No-social D042 calibration smoke — IMPLEMENTED, AWAITING IRIDIS VERIFICATION

New module:

    src/experiments/refined/calibration_smoke.py

Driver:

    scripts/run_refined_no_social_calibration_smoke.py

Tests:

    tests/test_refined_calibration_smoke.py

Smoke design:

    scale smoke seed       = 2026090204
    threshold smoke seed   = 2026090205
    scale smoke runs       = 3
    threshold smoke runs   = 3
    alpha                  = 0
    N                      = 100
    T                      = 1000
    B                      = 0
    L                      = 50
    rolling points/run     = 951

The two smoke namespaces are disjoint from final D042 namespaces `2026090201` and `2026090202`. Because smoke results are inspected, these paths must never be reused in the final 500+500 calibration.

Each no-social replication uses one canonical directed Random fixed-out-degree support only to keep the state/attention machinery valid. At `alpha=0`, graph/attention do not enter beliefs or market outcomes, so R/SW/SF are NOT replicated or ranked during calibration. This avoids pseudo-replicating one shock path three times.

The smoke validates end-to-end:

- frozen D043 baseline with only `alpha` replaced by zero;
- semantic shock, initial-state, and canonical graph seeds;
- 951 rolling component points per run;
- strictly positive pooled median `c_ret`, `c_bel`, and `c_F`;
- finite standardised CID paths;
- finite positive run-level peak CIDs and `c_CID`;
- baseline equal CID weights and inactive component guardrails;
- persistence of a smoke-only JSON artifact explicitly marked `final_calibration=false`.

New test file contributes 37 pytest cases.

Expected next checkpoint:

    567 passed

## Immediate gate

1. Pull latest `refined-model` on Iridis.
2. Run all refined tests; expected `567 passed`.
3. If green, run:

       python scripts/run_refined_no_social_calibration_smoke.py

4. Inspect the printed smoke scales/threshold and `results/refined/no_social_calibration_smoke/calibration_smoke.json` only for pipeline validity.
5. Do NOT freeze these 3+3 numerical values; they are smoke-only.
6. If the smoke succeeds, build the production D042 calibration runner/output layer.
7. Then run full 500 scale + 500 threshold no-social samples using only namespaces `2026090201` and `2026090202`.
8. Persist and freeze final numerical `c_ret`, `c_bel`, `c_F`, and `c_CID` before any confirmatory topology market experiment.

Large confirmatory topology-evaluation Monte Carlo remains prohibited until the final calibration artifact is frozen and a small paired confirmatory market runner has passed.

## Development status

    Phase 1  Refined fixed-topology core                         COMPLETE
    Phase 2  Topology generators                                COMPLETE
    Phase 3  Deterministic integration                          COMPLETE
    Phase 4  Paired design + structural validation              COMPLETE
    Phase 5  Market metrics / CID / calibration method          COMPLETE
    Phase 6  Mechanism diagnostics                              COMPLETE
    Phase 7  Frozen baseline + market calibration               IN PROGRESS
    Phase 8  Paired confirmatory market runner                  PLANNED
    Phase 9  alpha/beta/gamma experiments + heterogeneity       PLANNED
    Phase 10 Endogenous G formation                             PLANNED
    Phase 11 Full Jacobian / Lyapunov                           PLANNED
    Phase 12 State-space / EKF / empirical work                PLANNED
    Phase 13 Planner / policy                                   PLANNED
