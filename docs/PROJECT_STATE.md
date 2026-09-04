# Project State

Last updated: 2026-09-04

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

## Latest verified test checkpoint

Iridis:

    589 passed in 14.22s

with branch up to date and working tree clean.

This verifies the production D042 calibration runner, exact alpha=0 adaptive-vs-fixed attention equivalence for CID components, checkpoint/resume logic, specification fingerprints, and all earlier refined-model layers.

The subsequent Slurm-wrapper tests and newly added frozen-calibration tests are committed but await the next Iridis checkpoint.

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

Artifact:

    results/refined/no_social_calibration_smoke/calibration_smoke.json

It correctly records `final_calibration=false`. These 3+3 values are pipeline evidence only and MUST NOT be frozen or reused as D042 final calibration values.

## Production D042 calibration — COMPLETE

Production modules:

    src/experiments/refined/no_social_calibration_paths.py
    src/experiments/refined/market_calibration_run.py
    scripts/run_refined_market_calibration.py
    scripts/run_refined_market_calibration.slurm

Production run:

    scale replications      = 500 / 500
    threshold replications  = 500 / 500
    Slurm job               = 1505911
    compute host            = ruby047
    state                   = COMPLETED
    exit code               = 0
    elapsed                 = 01:06:36
    CPU efficiency          = 99.32%
    peak reported memory    = 218.15 MB

The production artifact is:

    results/refined/market_calibration/market_evaluation_calibration.json

It records `final_calibration=true`.

Final frozen numerical calibration:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

Reproducibility fingerprints:

    configuration = 9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202
    scales        = 1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476

Canonical frozen code API:

    src/experiments/refined/frozen_market_calibration.py
    first_frozen_market_evaluation_calibration()
    frozen_reference_scales()

Canonical documentation:

    docs/FINAL_MARKET_CALIBRATION.md

These numerical values are now fixed inputs to the first confirmatory topology experiments and must not be retuned after inspecting R/SW/SF outcomes.

## Calibration computational shortcut

At alpha=0, attention does not enter beliefs. Exact regression tests establish that adaptive and fixed attention generate identical return, belief, order-flow, and CID-component paths under common graph/shock/initial-state randomness.

The production D042 calibration therefore used `adaptive_attention=False` to avoid unnecessary attention updates. This shortcut applies only to D042 market/CID calibration outcomes, not to influence or attention diagnostics.

## Immediate gate

1. Pull latest `refined-model` on Iridis.
2. Run all refined tests. With four Slurm-wrapper tests plus six frozen-calibration tests added after the verified 589 checkpoint, expected total is:

       599 passed

3. If green, treat D042 numerical calibration as implementation-verified and immutable.
4. Build the paired confirmatory market-output runner using the frozen D043 market specification and `first_frozen_market_evaluation_calibration()`.
5. Run a small paired R/SW/SF confirmatory smoke with common random numbers and persistence checks.
6. Only after that small paired smoke passes may large confirmatory topology market Monte Carlo be submitted.

Large confirmatory topology-evaluation Monte Carlo remains prohibited until the paired confirmatory smoke passes.

## Development status

    Phase 1  Refined fixed-topology core                         COMPLETE
    Phase 2  Topology generators                                COMPLETE
    Phase 3  Deterministic integration                          COMPLETE
    Phase 4  Paired design + structural validation              COMPLETE
    Phase 5  Market metrics / CID / calibration method          COMPLETE
    Phase 6  Mechanism diagnostics                              COMPLETE
    Phase 7  Frozen baseline + market calibration               COMPLETE (pending 599-test verification of final freeze record)
    Phase 8  Paired confirmatory market runner                  NEXT
    Phase 9  alpha/beta/gamma experiments + heterogeneity       PLANNED
    Phase 10 Endogenous G formation                             PLANNED
    Phase 11 Full Jacobian / Lyapunov                           PLANNED
    Phase 12 State-space / EKF / empirical work                PLANNED
    Phase 13 Planner / policy                                   PLANNED
