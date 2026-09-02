# Project State

Last updated: 2026-09-02

## Project identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Branch:

    refined-model

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility only and never overrides the report.

Iridis shell setup:

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
- Eqs. (251)-(265): entropy/effective sources, realised influence/HHI, hub influence, overlap, mobility.

Eq. (266)-(267) KL-to-transition-prior remains deferred with attention inertia.

## Latest verified checkpoint

Iridis:

    568 passed in 14.28s

with branch up to date and working tree clean.

The corrected direct-script entry point is verified by a subprocess regression test.

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

Final numerical `c_ret`, `c_bel`, `c_F`, and `c_CID` do not yet exist.

## D043 first refined baseline — FROZEN AND VERIFIED

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

Frozen parameters:

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

W0 remains graph-supported uniform attention.

## sigma_0 evidence and report TODO

The CRN OAT sensitivity used:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}
    experiment seed = 2026090203
    5 paired replications

At the frozen `sigma_0=5e-4`, pooled medians were approximately:

    raw local reputation std / sigma_0    1.357
    mean attention mobility                0.02950
    max attention mobility                 0.32234
    final W distance from W0               0.29332
    return SD                              0.003428
    RMS mispricing                         0.068518
    projection fraction                    0.14385

REPORT TODO: add the complete sigma_0 sensitivity table to an Appendix / implementation-robustness section. Include at minimum reputation-dispersion ratio, mean/max attention mobility, final W distance, return SD, RMS mispricing, RMS flow per agent, desired-action p95, projection fraction, and inventory-boundary fraction for all five sigma_0 values. The main text should retain only the concise regularisation rationale.

## No-social D042 calibration smoke — VERIFIED

Smoke-only seeds:

    scale = 2026090204
    threshold = 2026090205

Design:

    3 scale runs + 3 threshold runs
    alpha = 0
    N = 100
    T = 1000
    B = 0
    L = 50
    rolling endpoints/run = 951

Observed smoke-only values:

    c_ret = 0.003087925449
    c_bel = 0.004183656828
    c_F   = 0.1172234222
    c_CID = 1.713734032

Threshold peak CIDs:

    1.672320382
    1.713734032
    1.386226693

Artifact:

    results/refined/no_social_calibration_smoke/calibration_smoke.json

It correctly records `final_calibration=false`. These 3+3 values are pipeline evidence only and MUST NOT be frozen or reused as D042 final calibration values.

## Production D042 calibration runner — IMPLEMENTED, AWAITING IRIDIS VERIFICATION

New shared path module:

    src/experiments/refined/no_social_calibration_paths.py

Production runner:

    src/experiments/refined/market_calibration_run.py

CLI:

    scripts/run_refined_market_calibration.py

Tests:

    tests/test_refined_market_calibration_run.py

Production design:

- exact D042 500 scale + 500 threshold counts by default;
- exact D042 seed namespaces 2026090201 / 2026090202;
- one canonical valid graph per no-social replication, not R/SW/SF triplets;
- scale replications checkpoint to compressed NPZ files;
- threshold replications checkpoint to JSON peak-CID files;
- every checkpoint is bound to a SHA-256 fingerprint of the full D042 protocol + D043 baseline;
- threshold checkpoints are also bound to the realised reference-scale fingerprint;
- stages are resumable after interruption;
- scale artifact is marked stage-complete but `final_calibration=false`;
- final artifact is written only after all threshold replications complete;
- CLI refuses production execution on a hostname containing `login` unless deliberately overridden.

Computational shortcut:

At alpha=0, attention does not enter beliefs. Therefore adaptive and fixed attention must generate exactly identical return/belief/order-flow/CID-component paths under common graph/shock/initial-state randomness. The production runner uses `adaptive_attention=False` only after this exact equivalence is regression-tested. This shortcut applies to D042 calibration outcomes only, not to influence/attention diagnostics.

New production test file contributes 21 cases.

Expected next checkpoint:

    589 passed

## Immediate gate

1. Pull latest `refined-model` on Iridis.
2. Run all refined tests; expected `589 passed`.
3. Do NOT run `scripts/run_refined_market_calibration.py` directly on `loginX...`.
4. If tests pass, prepare/submit the production calibration inside an Iridis compute allocation/batch job.
5. The runner is resumable; interrupted completed replications are reused only when their specification fingerprints match exactly.
6. After the full 500+500 artifact is produced, inspect and freeze final `c_ret`, `c_bel`, `c_F`, and `c_CID` before any confirmatory topology market experiment.
7. Then build the paired confirmatory market-output runner and small paired smoke.

Large confirmatory topology-evaluation Monte Carlo remains prohibited until the final calibration artifact is frozen and the paired confirmatory smoke passes.

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
