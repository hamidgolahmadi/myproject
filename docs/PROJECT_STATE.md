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

    528 passed in 8.31s

with clean working tree and branch up to date with `origin/refined-model`.

This verifies:

- D042 separate-sample market-evaluation calibration METHOD;
- provisional baseline specification / neutral initialisation machinery;
- topology-blind market-scale smoke diagnostics;
- common-random-number OAT sigma_0 sensitivity machinery.

The D043 freeze changes described below are newly committed and await the next Iridis test checkpoint.

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

## D043 first refined baseline — FROZEN

Canonical module:

    src/experiments/refined/baseline_specification.py

Canonical documentation:

    docs/REFINED_BASELINE.md

Archived pre-freeze note:

    docs/REFINED_BASELINE_CANDIDATE.md

Canonical API:

    RefinedBaselineSpecification
    first_refined_baseline_specification()

Compatibility aliases retained:

    RefinedBaselineCandidate
    first_refined_baseline_candidate()

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

## Evidence for D043 freeze

The first topology-blind scale smoke used:

    experiment_seed = 2026090203
    5 paired replications
    3 topology treatments per replication
    N = 100
    T = 1000

It found finite, non-degenerate market dynamics.  With the initial candidate:

    median return SD                      0.00342286
    median RMS mispricing                 0.0685422
    median RMS flow per agent             0.101498
    median |desired action| p95           0.304871
    desired-action saturation fraction   0
    median projection fraction            0.14608
    median inventory-boundary fraction    0.14608

The only unresolved scale issue was the reputation-dispersion floor `sigma_0=1e-6`.

A common-random-number OAT sensitivity then compared:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

using the same graph realisations, shock paths, initial states, and all other parameters.

Return volatility, mispricing, order flow, desired-action scale, and inventory projection were essentially invariant across the grid.  The intended attention regularisation changed smoothly.

At the frozen value `sigma_0=5e-4`, pooled medians were approximately:

    raw local reputation std / sigma_0    1.357
    mean attention mobility                0.02950
    max attention mobility                 0.32234
    final W distance from W0               0.29332
    return SD                              0.003428
    RMS mispricing                         0.068518
    projection fraction                    0.14385

The value is frozen because the floor is of the same order as realised local reputation dispersion: it meaningfully regularises near-degenerate local reputation distributions without dominating them or shutting down adaptive attention.  Selection used pooled absolute diagnostics only, never topology rankings.

## Report revision TODO — sigma_0 sensitivity appendix

When the doctoral report is next revised, add the full pre-freeze `sigma_0` sensitivity table to an Appendix / implementation-robustness section.

Required grid:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

The Appendix table should report, at minimum, for each `sigma_0`:

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

Use the already completed common-random-number sensitivity run with experiment seed `2026090203` and 5 paired replications. The table is a regularisation-sensitivity diagnostic, not a topology-ranking table.

The main text should retain only a concise explanation of why `sigma_0=5e-4` was chosen; the complete numerical sensitivity evidence belongs in the Appendix.

## Immediate gate

New D043 implementation/tests are committed but not yet verified on Iridis.

Next checkpoint:

    python -m pytest -q tests/test_refined_*.py

Expected total:

    530 passed

If 530 passes, D043 implementation is VERIFIED.

Then the next development task is a SMALL end-to-end no-social D042 calibration smoke under the frozen D043 baseline.  That smoke should validate:

- `alpha=0` is enforced in the calibration market specification;
- scale and threshold seed namespaces are distinct;
- rolling component paths have the expected 951 endpoints for T=1000, B=0, L=50;
- the pooled scale medians are strictly positive;
- standardised CID paths and run-level peak CID are finite;
- persistence/serialization of a calibration artifact works;
- no R/SW/SF ranking is inspected or used.

Only after that small no-social calibration smoke succeeds may the full D042 500+500 calibration be run.

Large confirmatory topology-evaluation Monte Carlo remains prohibited until the numerical calibration artifact is frozen and a small paired confirmatory market runner has passed.

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
