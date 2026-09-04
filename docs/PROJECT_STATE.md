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

    660 passed in 15.50s

This verifies the complete refined core, paired seed/treatment machinery,
D041 structural validation, market/CID/mechanism diagnostics, D042/D044 market
evaluation calibration, D043 baseline, Phase-8 paired runner/smoke, and the
complete D045 production/inference layer.

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

Production calibration Slurm job 1505911 completed on ruby047 with exit code 0.

Frozen values:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

Fingerprints:

    configuration = 9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202
    scales        = 1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476

These values remain fixed for D046; alpha sweeps do not recalibrate CID.

## Phase 8 paired confirmatory smoke — VERIFIED

Smoke namespace:

    2026090401

Design:

    2 paired replications
    R / SW / SF
    baseline alpha=0.75 + alpha0 control
    12 simulations

Iridis job 1509863 completed on ruby047 with exit code 0 in 00:01:50.

Exact end-to-end alpha-zero topology-null control PASSED:

    replication 0: 1 unique economic-path fingerprint across R/SW/SF
    replication 1: 1 unique economic-path fingerprint across R/SW/SF

The smoke is pipeline evidence only and is not used to rank topologies.

## D045 first confirmatory production — VERIFIED AND FROZEN

Canonical protocol/code:

    src/experiments/refined/confirmatory_protocol.py
    src/experiments/refined/confirmatory_inference.py
    src/experiments/refined/confirmatory_production.py
    docs/D045_CONFIRMATORY_PROTOCOL.md
    docs/D045_RESULTS.md

Frozen production design:

    baseline alpha = 0.75
    production seed = 2026090402
    paired replications = 1000
    topology triplet = (R, SW, SF)
    simulations = 3000
    bootstrap seed = 2026090403
    bootstrap draws = 10000
    confidence level = 95%
    family-wise alpha = 0.05

Production provenance:

    Slurm array = 1511972
    10/10 tasks COMPLETED
    every task exit code = 0:0
    stderr = empty
    production/finalizer commit = b5fbf52dd988637d90d7b5bc5c346c20551b66be
    finalization job = 1512116, COMPLETED exit code 0

Final artifacts:

    results/refined/confirmatory_production/confirmatory_records.csv
    results/refined/confirmatory_production/confirmatory_metadata.json
    results/refined/confirmatory_production/confirmatory_analysis.json
    results/refined/confirmatory_production/topology_means.csv
    results/refined/confirmatory_production/topology_gaps.csv
    results/refined/confirmatory_production/pairwise_contrasts.csv

Primary result summary:

- return volatility: `SF > R > SW`; all three pairwise contrasts survive Holm;
- mean absolute order flow/agent: `SF > R > SW`; all three survive Holm;
- peak CID: SW is lower than R and SF; R-SF does not survive Holm;
- RMS mispricing: no pair survives Holm;
- maximum absolute mispricing: no pair survives Holm;
- threshold-exceeding rate is 0.1%-0.2% and no pair survives Holm.

Mechanism result summary:

    mean hub influence share:       SF > R > SW
    mean attention overlap:         SF > R > SW
    mean pairwise action covariance:SF > R > SW
    aggregate order-flow variance:  SF > R > SW

All 12 predeclared mechanism contrasts survive the separate Holm correction.
The effect attenuates strongly along the chain from realised social influence to
final price instability. Therefore D045 supports a strong network-transmission
mechanism but only modest realised instability differences at alpha=0.75.

## D046 exploratory alpha sweep — FROZEN, IMPLEMENTED, AWAITING TEST VERIFICATION

Scientific purpose:

Map where topology differentiation is weak, strongest, or saturated as alpha
changes, without changing any other D043 parameter or D044 evaluation scale.
This is explicitly OAT/exploratory under D027, not a second confirmatory family.

Canonical protocol/documentation:

    src/experiments/refined/alpha_sweep_protocol.py
    docs/D046_ALPHA_SWEEP_PROTOCOL.md

Frozen design:

    experiment seed = 2026090404
    alpha grid = (0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 1.00)
    paired replications per alpha = 300
    topology triplet = (R, SW, SF)
    total simulations = 7200
    bootstrap seed = 2026090405
    bootstrap draws = 5000
    confidence level = 95%

Within replication, the same semantic shock/initial-state/graph seeds are used
across the full alpha grid. Alpha does not enter the exogenous shock scale or
neutral initial-state distribution, and a dedicated regression test locks this
cross-alpha CRN property. Parameter fingerprints still differ by alpha and are
validated normally.

One bootstrap unit is the complete replication block containing all 8 alpha
values and all 3 topology treatments. Independent resampling by alpha or by
topology is prohibited.

D046 analysis reports, at each alpha:

    topology means
    absolute/relative topology gaps
    R-SW, R-SF, SW-SF paired contrasts
    95% matched-block percentile intervals

No Holm/FWER rejection family is attached to D046. Final analysis requires the
exact alpha=0 economic-path topology null in every replication.

Implementation:

    src/experiments/refined/alpha_sweep_analysis.py
    src/experiments/refined/alpha_sweep_production.py
    scripts/run_refined_alpha_sweep.py
    scripts/run_refined_alpha_sweep.slurm
    scripts/finalize_refined_alpha_sweep.py
    scripts/finalize_refined_alpha_sweep.slurm

Checkpoint:

    results/refined/alpha_sweep/checkpoints/alpha_XX/replication_XXXX.json

Finalization is impossible until all `8 x 300 = 2400` alpha/replication
checkpoints validate.

Slurm design:

    48 array tasks
    8 alpha slices x 6 blocks
    50 paired replications per task
    maximum 16 concurrent tasks
    1 CPU/task, 4 GB/task, 1 hour/task

28 new tests were added after the verified 660 checkpoint.

Expected next checkpoint:

    688 passed

## Report revision TODO — sigma_0 Appendix

Add the complete CRN sensitivity table for `sigma_0={1e-6,1e-4,5e-4,1e-3,2e-3}`
using experiment seed 2026090203 and the completed 5 paired replications. This is
a regularisation-sensitivity diagnostic, not a topology-ranking table.

## Immediate gate

1. Pull latest `refined-model` on Iridis.
2. Run all refined tests; expected `688 passed`.
3. Confirm working tree clean.
4. Do NOT submit D046 until this gate is green.
5. If green, create `results/refined/alpha_sweep` before `sbatch` so Slurm can open logs.
6. Do not inspect partial alpha/topology curves before all 2400 checkpoints exist.

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
    Phase 9   alpha/beta/gamma experiments + heterogeneity       IN PROGRESS (D046 alpha)
    Phase 10  Endogenous G formation                             PLANNED
    Phase 11  Full Jacobian / Lyapunov                           PLANNED
    Phase 12  State-space / EKF / empirical work                 PLANNED
    Phase 13  Planner / policy                                   PLANNED
