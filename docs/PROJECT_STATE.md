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

    619 passed in 15.76s

with branch up to date and working tree clean.

This verifies the complete refined core, paired seed/treatment machinery,
D041 structural validation, market/CID/mechanism diagnostics, D042 production
calibration runner, D043 baseline, D044 frozen numerical calibration, and the
Phase-8 paired confirmatory runner/smoke.

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

These values are immutable for the first confirmatory topology experiment.

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

## D045 first confirmatory production protocol — FROZEN, IMPLEMENTED, AWAITING VERIFICATION

Canonical protocol:

    src/experiments/refined/confirmatory_protocol.py
    docs/D045_CONFIRMATORY_PROTOCOL.md

Frozen production design:

    production seed = 2026090402
    paired replications = 1000
    topology triplet = (R, SW, SF)
    simulations = 3000
    bootstrap seed = 2026090403
    bootstrap draws = 10000
    confidence level = 95%
    family-wise alpha = 0.05

The first production run is baseline-only (`alpha=0.75`). The exact alpha-zero
negative control is not repeated 1000 times because its market-path topology
null has already been established exactly and verified end-to-end in Phase 8.

Predeclared pairwise contrasts:

    R - SW
    R - SF
    SW - SF

Primary confirmatory family, Holm FWER over 18 pairwise hypotheses:

    return_volatility
    rms_mispricing
    maximum_absolute_mispricing
    mean_absolute_order_flow_per_agent
    peak_cid
    threshold_exceeding

Mechanism confirmatory family, separate Holm FWER over 12 hypotheses:

    mean_hub_influence_share
    mean_attention_overlap
    mean_pairwise_action_covariance
    mean_aggregate_order_flow_variance

Secondary outcomes receive pointwise exploratory bootstrap intervals. The
right-censoring/non-stabilisation rate is retained and censored runs are never
dropped.

Bootstrap resamples complete matched replication triplets, never topology
samples independently.

## D045 implementation

New/updated modules:

    src/experiments/refined/paired.py
    src/experiments/refined/treatments.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/confirmatory_runner.py
    src/experiments/refined/confirmatory_protocol.py
    src/experiments/refined/confirmatory_inference.py
    src/experiments/refined/confirmatory_production.py

Key guards:

- paired plans now bind `n_agents`, `n_periods`, and the exact parameter-vector SHA-256 fingerprint;
- treatment preparation rejects reuse of a shock plan with different parameters;
- rolling action covariance is algebraically equivalent but vectorised for production efficiency;
- every production checkpoint is one indivisible R/SW/SF triplet;
- checkpoints are specification-fingerprinted and resumable;
- final inference artifacts are impossible until all 1000 checkpoints exist;
- no production task prints or ranks topology contrasts before finalization.

Production CLI / Slurm:

    scripts/run_refined_confirmatory_production.py
    scripts/run_refined_confirmatory_production.slurm
    scripts/finalize_refined_confirmatory_production.py
    scripts/finalize_refined_confirmatory_production.slurm

Array design:

    10 tasks x 100 paired replications
    1 CPU/task
    4 GB/task
    2 hour task walltime

No partition/account was invented; Iridis default scheduling is retained.

Final artifacts, only after full completion:

    results/refined/confirmatory_production/confirmatory_records.csv
    results/refined/confirmatory_production/confirmatory_metadata.json
    results/refined/confirmatory_production/confirmatory_analysis.json
    results/refined/confirmatory_production/topology_means.csv
    results/refined/confirmatory_production/topology_gaps.csv
    results/refined/confirmatory_production/pairwise_contrasts.csv

## D045 test gate

New regression tests cover:

- parameter-plan binding;
- vectorised Eq. (239)-(240) equivalence;
- frozen D045 protocol;
- matched-triplet bootstrap and inference;
- checkpoint/resume/finalization;
- CLI and Slurm array/finalizer contracts.

41 tests were added after the verified 619 checkpoint.

Expected next checkpoint:

    660 passed

Do NOT submit the D045 production array until this checkpoint is verified on
Iridis with a clean working tree.

## Report revision TODO — sigma_0 Appendix

Add the complete CRN sensitivity table for `sigma_0={1e-6,1e-4,5e-4,1e-3,2e-3}`
using experiment seed 2026090203 and the completed 5 paired replications. This is
a regularisation-sensitivity diagnostic, not a topology-ranking table.

## Immediate gate

1. Pull latest `refined-model` on Iridis.
2. Run all refined tests; expected `660 passed`.
3. Confirm working tree clean.
4. Only if green, create `results/refined/confirmatory_production` and submit the D045 Slurm array.
5. Do not run finalization until all 1000 paired checkpoints exist.
6. Do not inspect or interpret partial topology contrasts during production.

## Development status

    Phase 1  Refined fixed-topology core                         COMPLETE
    Phase 2  Topology generators                                COMPLETE
    Phase 3  Deterministic integration                          COMPLETE
    Phase 4  Paired design + structural validation              COMPLETE
    Phase 5  Market metrics / CID / calibration method          COMPLETE
    Phase 6  Mechanism diagnostics                              COMPLETE
    Phase 7  Frozen baseline + market calibration               COMPLETE
    Phase 8  Paired confirmatory market runner                  COMPLETE
    Phase 9  D045 confirmatory production / large MC            IN PROGRESS
    Phase 10 alpha/beta/gamma experiments + heterogeneity       PLANNED
    Phase 11 Endogenous G formation                             PLANNED
    Phase 12 Full Jacobian / Lyapunov                           PLANNED
    Phase 13 State-space / EKF / empirical work                PLANNED
    Phase 14 Planner / policy                                   PLANNED
