# Project State

Last updated: 2026-09-07

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

## Latest verified code checkpoint

Iridis:

    689 passed in 16.39s
    working tree clean

This verifies the complete refined core, paired seed/treatment machinery,
D041 structural validation, D042/D044 market evaluation, D043 baseline, D045
confirmatory production/inference, and the complete D046 alpha-sweep code layer.

D047 code has now been added after this verified checkpoint and is awaiting the
next Iridis test gate.

## Frozen structural design — D041

    N=100, K=6, q=5, p_sw=0.02, a0=1.0

1000 graph replications per topology passed the structural gate. Ensemble means:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

## Frozen market baseline — D043

    N=100, K=6, T=1000, q=5, p_sw=0.02, a0=1.0
    rho_theta=0.985
    sigma_theta=0.025
    v_bar=0.0
    psi=1.0
    sigma_s=0.06
    sigma_b=0.025
    alpha=0.75
    kappa=2.4
    x_bar=5.0
    chi=0.02
    lambda_price=0.0002
    sigma_p=0.001
    gamma_R=0.9
    beta=1.0
    sigma_0=0.0005

Neutral non-network initialisation:

    theta_0 ~ stationary AR(1)
    b_i,0 = theta_0
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

W0 remains topology-specific uniform graph-supported attention.

## Frozen market evaluation — D042 + D044

    B=0
    rolling L=50
    robustness L={25,100}
    scale sample=500, seed 2026090201
    threshold sample=500, seed 2026090202
    equal CID weights
    c_CID=95th percentile of run-level peak CID, method=higher
    component guardrails inactive
    L_stab=50

Frozen values:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

Fingerprints:

    configuration = 9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202
    scales        = 1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476

These values remain fixed throughout D046 and D047; behavioural sweeps do not
recalibrate CID.

## D045 first confirmatory production — VERIFIED AND FROZEN

Canonical result summary:

    docs/D045_RESULTS.md

Production:

    alpha = 0.75
    seed = 2026090402
    paired replications = 1000
    simulations = 3000
    bootstrap seed = 2026090403
    bootstrap draws = 10000
    Slurm array = 1511972
    finalizer = 1512116
    production/finalizer commit = b5fbf52dd988637d90d7b5bc5c346c20551b66be

Primary result: return volatility and mean absolute order flow show the small but
statistically detectable ordering SF > R > SW; peak CID places SW below R/SF;
mispricing and threshold-exceeding do not show robust family-wise topology
separation.

Mechanism result: all 12 predeclared mechanism contrasts survive the separate
Holm correction, with strong SF > R > SW ordering for hub influence, attention
overlap, pairwise action covariance, and aggregate order-flow variance.

Interpretation: topology strongly reorganises the social transmission mechanism,
but under the frozen baseline its realised price-instability effect is modest.

## D046 exploratory alpha sweep — VERIFIED AND COMPLETE

Canonical protocol/results:

    src/experiments/refined/alpha_sweep_protocol.py
    src/experiments/refined/alpha_sweep_analysis.py
    src/experiments/refined/alpha_sweep_production.py
    docs/D046_ALPHA_SWEEP_PROTOCOL.md
    docs/D046_RESULTS.md

Frozen design:

    experiment seed = 2026090404
    alpha grid = (0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 0.99)
    paired replications per alpha = 300
    total simulations = 7200
    bootstrap seed = 2026090405
    bootstrap draws = 5000

The original draft alpha=1 endpoint was rejected before production because the
report and RefinedParameters require 0 <= alpha < 1. It was replaced by 0.99
before D046 outcomes were inspected.

Execution provenance:

    production Slurm array = 1526697
    48/48 tasks COMPLETED, every task exit code 0:0
    successful completion markers = 48
    non-empty stderr files = 0
    checkpoints = 2400/2400
    finalization job = 1527139, COMPLETED exit code 0
    finalizer host = ruby047
    production/finalizer commit = 5283a338d75560f27d28a19d48e07f86cdeada07

Final D046 validation:

    alpha_zero_economic_path_null_verified = True
    n_replications = 300
    n_bootstrap = 5000

Main D046 result:

- absolute return volatility, order-flow activity, CID, and aggregate order-flow
  variance fall as alpha rises;
- cross-topology differentiation nevertheless strengthens from low alpha into
  the coherent pre-boundary region;
- through roughly alpha=0.85 the key market/mechanism ordering is generally
  SF > R > SW;
- alpha=0.95--0.99 displays ranking reversals and a qualitatively different
  high-social-weight boundary regime;
- structural hub influence itself remains strongly SF > R > SW across the grid;
- threshold-exceeding is zero for all topologies from alpha=0.40 onward in the
  D046 sample.

D046 therefore supports a level/gap distinction: stronger social weight can
reduce absolute market activity while increasing topology differentiation.

## D047 exploratory beta sweep — FROZEN AND IMPLEMENTED, AWAITING TEST VERIFICATION

Scientific purpose:

Map how reputational selectivity activates or saturates the topology mechanism
at a fixed D046-selected social-weight anchor.

Canonical protocol/documentation:

    src/experiments/refined/beta_sweep_protocol.py
    docs/D047_BETA_SWEEP_PROTOCOL.md

Frozen design:

    experiment seed = 2026090701
    alpha anchor = 0.85
    beta grid = (0.00, 0.01, 0.10, 0.50, 1.00, 2.00, 5.00, 10.00, 100.00, 1000.00)
    paired replications per beta = 300
    topology triplet = (R, SW, SF)
    total simulations = 9000
    bootstrap seed = 2026090702
    bootstrap draws = 5000
    confidence level = 95%

Rationale:

- alpha=0.85 is selected after D046 as the strongest coherent pre-boundary
  anchor; D047 is therefore exploratory, not confirmatory;
- beta=0 is an exact no-selectivity/uniform-attention control;
- beta=1 retains the D043 baseline anchor;
- beta=2 and 5 resolve the report's pilot transition region;
- beta up to 1000 retains the report-scale logarithmic high-selectivity range.

Within replication, the same semantic shock path, neutral initial state, and
R/SW/SF graph seeds are preserved across the full beta grid. Parameter
fingerprints still change with beta. One bootstrap unit is the complete
10-beta x 3-topology replication block.

Implementation:

    src/experiments/refined/beta_sweep_analysis.py
    src/experiments/refined/beta_sweep_production.py
    scripts/run_refined_beta_sweep.py
    scripts/run_refined_beta_sweep.slurm
    scripts/finalize_refined_beta_sweep.py
    scripts/finalize_refined_beta_sweep.slurm

D047 deliberately wraps the unchanged ConfirmatoryTreatmentRecord rather than
adding beta to that frozen D045/D046 schema. This preserves reproducibility of
older checkpoint/configuration fingerprints while giving D047 a flat beta-tagged
CSV schema of its own.

Checkpoint path:

    results/refined/beta_sweep/checkpoints/beta_XX/replication_XXXX.json

Finalization requires all 10 x 300 = 3000 beta/replication checkpoints.

Slurm design:

    60 array tasks
    10 beta slices x 6 blocks
    50 paired replications per task
    maximum 16 concurrent tasks
    1 CPU/task, 4 GB/task, 1 hour/task

31 new D047 tests have been added after the verified 689 checkpoint.

Expected next checkpoint:

    720 passed

## Report revision TODO — sigma_0 Appendix

Add the complete CRN sensitivity table for sigma_0={1e-6,1e-4,5e-4,1e-3,2e-3}
using experiment seed 2026090203 and the completed five paired replications. This
is a regularisation-sensitivity diagnostic, not a topology-ranking table.

## Immediate gate

1. Pull latest `refined-model` on Iridis.
2. Run all refined tests; expected `720 passed`.
3. Confirm working tree clean.
4. Do NOT submit D047 until this gate is green.
5. If green, create `results/refined/beta_sweep` before `sbatch`.
6. Do not inspect partial beta/topology curves; finalization requires all 3000 checkpoints.

## Development status

    Phase 1   Refined fixed-topology core                         COMPLETE
    Phase 2   Topology generators                                COMPLETE
    Phase 3   Deterministic integration                          COMPLETE
    Phase 4   Paired design + structural validation              COMPLETE
    Phase 5   Market metrics / CID / calibration method          COMPLETE
    Phase 6   Mechanism diagnostics                              COMPLETE
    Phase 7   Frozen baseline + market calibration               COMPLETE
    Phase 8   Paired confirmatory market runner                  COMPLETE
    Phase 8b  D045 confirmatory production / large MC            COMPLETE
    Phase 9a  D046 alpha sweep                                   COMPLETE
    Phase 9b  D047 beta sweep                                    IN PROGRESS
    Phase 9c  gamma_R experiments + heterogeneity                PLANNED
    Phase 10  Endogenous G formation                             PLANNED
    Phase 11  Full Jacobian / Lyapunov                           PLANNED
    Phase 12  State-space / EKF / empirical work                 PLANNED
    Phase 13  Planner / policy                                   PLANNED
