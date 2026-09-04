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

## Latest verified checkpoint

Iridis:

    599 passed in 11.64s

with branch up to date and working tree clean.

This verifies the complete refined core, paired seed/treatment machinery,
D041 structural validation, market/CID/mechanism diagnostics, D042 production
calibration runner, Slurm wrapper, D043 frozen baseline, and D044 frozen
numerical market calibration.

## Frozen structural design — D041

    N = 100
    K = 6
    q = 5
    p_sw = 0.02
    a0 = 1.0

1000 graph replications per topology passed the structural gate. Ensemble means:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

## Frozen market baseline — D043

    N = 100
    K = 6
    T = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

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

W0 remains topology-specific uniform graph-supported attention.

## Frozen market evaluation — D042 + D044

Method:

    B = 0
    rolling L = 50
    robustness L = {25, 100}
    alpha_calibration = 0
    scale sample = 500 runs, namespace 2026090201
    threshold sample = 500 separate runs, namespace 2026090202
    CID weights = equal thirds
    c_CID = 95th percentile of run-level peak CID, method=higher
    component guardrails = inactive
    L_stab = 50

Production Slurm job 1505911 completed on ruby047 with exit code 0.

Frozen numerical calibration:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

Fingerprints:

    configuration = 9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202
    scales        = 1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476

Canonical API:

    first_frozen_market_evaluation_calibration()
    frozen_reference_scales()

These values are immutable for the first confirmatory topology experiments.

## Report revision TODO — sigma_0 Appendix

Add the complete CRN sensitivity table for:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

using experiment seed 2026090203 and the completed 5 paired replications.
Include reputation-dispersion ratio, mean/max attention mobility, final W
distance, return SD, RMS mispricing, RMS flow per agent, desired-action p95,
projection fraction, and inventory-boundary fraction. The table is a
regularisation-sensitivity diagnostic, not a topology-ranking table.

## Phase 8 — paired confirmatory runner — IMPLEMENTED, AWAITING VERIFICATION

New core module:

    src/experiments/refined/confirmatory_runner.py

Driver:

    scripts/run_refined_confirmatory_smoke.py

Tests:

    tests/test_refined_confirmatory_runner.py

Canonical runner:

    run_paired_confirmatory_replication(...)

For each replication it:

1. prepares one common shock path and common non-network initial state;
2. generates topology-specific R/SW/SF graph seeds, G, and W0(G);
3. runs the frozen D043 model with adaptive attention;
4. evaluates the frozen D044 CID calibration;
5. records run-level market outcomes;
6. records CID peak/exceedance/duration/stabilisation;
7. records structural graph diagnostics;
8. records time-averaged realised-influence diagnostics;
9. records semantic seeds and a SHA-256 fingerprint of the complete non-attention economic path.

The runner does not estimate treatment effects or rank topologies.

## Confirmatory smoke design

Smoke-only namespace:

    experiment_seed = 2026090401

Default smoke:

    2 paired replications
    3 topology treatments: R, SW, SF
    2 regimes: baseline alpha=0.75 and alpha0 negative control
    total simulations = 12

The baseline and alpha0 regimes reuse the same semantic seed namespace and
replication IDs. Since shock and neutral-initial-state laws do not depend on
alpha, this preserves common randomness across the control contrast as well.

Critical end-to-end negative control:

    alpha = 0

must produce exactly one unique economic-path fingerprint across R/SW/SF within
each replication, even though graph and influence diagnostics remain
 topology-specific. This validates that network propagation is truly absent
from market outcomes while the structural treatment objects remain distinct.

Smoke persistence:

    results/refined/confirmatory_smoke/confirmatory_smoke_records.csv
    results/refined/confirmatory_smoke/confirmatory_smoke_metadata.json

Metadata is explicitly:

    final_confirmatory = false
    interpretation_guard = do not rank topologies from this smoke

No qualitative topology ordering is asserted in unit tests. D041 already
validated treatment architecture at the ensemble level.

The new confirmatory test file contributes 20 cases.

Expected next checkpoint:

    619 passed

## Immediate gate

1. Pull latest `refined-model` on Iridis.
2. Run all refined tests; expected `619 passed`.
3. If green, run the paired confirmatory smoke.
4. Verify for every alpha0 replication that R/SW/SF produce exactly one unique economic-path fingerprint.
5. Verify CSV/JSON persistence and `final_confirmatory=false`.
6. Do not interpret or rank baseline topology outcomes from the smoke.
7. Only after the smoke passes should we design the resumable large confirmatory Monte Carlo output layer and freeze its experiment seed/sample size.

Large confirmatory topology Monte Carlo remains prohibited until this smoke passes.

## Development status

    Phase 1  Refined fixed-topology core                         COMPLETE
    Phase 2  Topology generators                                COMPLETE
    Phase 3  Deterministic integration                          COMPLETE
    Phase 4  Paired design + structural validation              COMPLETE
    Phase 5  Market metrics / CID / calibration method          COMPLETE
    Phase 6  Mechanism diagnostics                              COMPLETE
    Phase 7  Frozen baseline + market calibration               COMPLETE
    Phase 8  Paired confirmatory market runner                  IN PROGRESS
    Phase 9  alpha/beta/gamma experiments + heterogeneity       PLANNED
    Phase 10 Endogenous G formation                             PLANNED
    Phase 11 Full Jacobian / Lyapunov                           PLANNED
    Phase 12 State-space / EKF / empirical work                PLANNED
    Phase 13 Planner / policy                                   PLANNED
