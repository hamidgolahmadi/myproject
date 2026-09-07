# Refined Model Implementation Map

Last updated: 2026-09-07

Scientific source of truth: `report1_25_08_2026.pdf`. Legacy code is reference only.

## Core runtime — Eqs. (35)-(82)

    src/model/refined/state.py
    src/model/refined/shocks.py
    src/model/refined/fundamentals.py
    src/model/refined/beliefs.py
    src/model/refined/trading.py
    src/model/refined/market.py
    src/model/refined/reputation.py
    src/model/refined/attention.py
    src/model/refined/transition.py
    src/model/refined/simulator.py

Canonical timing:

    W_{t-1}
      -> theta_t, v_t, s_t
      -> b_t
      -> vhat_t -> m_t -> desired action -> executed action
      -> x_t -> F_t -> p_t -> r_t -> pi_t -> R_t -> z_t -> W_t

Status: VERIFIED.

## Paired design and semantic randomness

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py
    src/experiments/refined/treatments.py

Common within paired replication:

    shock path
    non-network initial state
    parameters
    horizon
    evaluation definitions

Topology-specific:

    graph seed
    realised G
    graph-supported W0

`PairedReplicationPlan` stores n_agents, n_periods, the exact parameter
fingerprint, semantic seeds, topology graph seeds, and the common shock path.
Treatment construction validates the parameter fingerprint before execution.

Status: VERIFIED.

## Topologies and structural validation — D041

    src/topologies/refined/generators.py
    src/topologies/refined/diagnostics.py
    src/experiments/refined/structural.py
    src/experiments/refined/calibration.py
    src/experiments/refined/structural_io.py
    scripts/run_refined_structural_validation.py

Matched design:

    N=100, K=6, q=5, p_sw=0.02, a0=1.0

1000 graphs per R/SW/SF topology passed the structural gate.

Status: VERIFIED.

## Market outcomes and CID

    src/experiments/refined/market_metrics.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/cid.py
    src/experiments/refined/cid_events.py

Mapping:

    Eqs. (231)-(235): evaluation sample / rolling windows
    Eqs. (236)-(238): RV, RMSM, MAM, MAF
    Eqs. (239)-(240): rolling action covariance / Var(F) decomposition
    Eqs. (241)-(246): CID components / scales / weights / CID
    Eqs. (247)-(250): exceedance / duration / stabilisation / censoring
    Eqs. (288)-(289): MAR / time-averaged belief variance

`rolling_action_covariance` uses the exact Eq. (240) variance decomposition rather
than constructing an N-by-N covariance matrix at every rolling endpoint.

Status: VERIFIED.

## Realised influence — Eqs. (251)-(265)

    src/experiments/refined/influence_metrics.py

Implements normalised entropy, effective sources, source shares, influence HHI,
structural-hub realised influence, attention overlap, and attention mobility.
Structural hubs come from directed in-degree in G, never W_t.

Status: VERIFIED.

Eqs. (266)-(267) KL-to-transition-prior remain deferred with attention inertia.

## D043 frozen baseline

    src/experiments/refined/baseline_specification.py
    docs/REFINED_BASELINE.md

    N=100, K=6, T=1000, q=5, p_sw=0.02, a0=1.0
    alpha=0.75, kappa=2.4, chi=0.02, lambda_price=0.0002
    gamma_R=0.9, beta=1.0, sigma_0=0.0005

Status: VERIFIED.

## D042 / D044 frozen market evaluation

    src/experiments/refined/market_calibration.py
    src/experiments/refined/no_social_calibration_paths.py
    src/experiments/refined/market_calibration_run.py
    src/experiments/refined/frozen_market_calibration.py
    docs/FINAL_MARKET_CALIBRATION.md

Frozen values:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

Production calibration job 1505911 completed 500+500 no-social runs.

Status: VERIFIED.

## Paired confirmatory market runner — Phase 8

    src/experiments/refined/confirmatory_runner.py

Canonical API:

    ConfirmatoryTreatmentRecord
    ConfirmatorySmokeResult
    run_paired_confirmatory_replication(...)
    run_paired_confirmatory_smoke(...)
    write_paired_confirmatory_smoke(...)

The runner is intentionally retained unchanged as the common treatment engine
for later sweeps. D047 does not add beta to `ConfirmatoryTreatmentRecord`; it
wraps that frozen schema in `BetaSweepTreatmentRecord`, preserving D045/D046
checkpoint/configuration reproducibility.

Status: VERIFIED.

## D045 confirmatory production — VERIFIED AND FROZEN

Protocol/inference/production:

    src/experiments/refined/confirmatory_protocol.py
    src/experiments/refined/confirmatory_inference.py
    src/experiments/refined/confirmatory_production.py
    docs/D045_CONFIRMATORY_PROTOCOL.md
    docs/D045_RESULTS.md

Production array 1511972 completed 1000 matched triplets on commit
`b5fbf52dd988637d90d7b5bc5c346c20551b66be`; finalizer 1512116 completed on
the same commit.

Status: VERIFIED AND FROZEN.

## D046 exploratory alpha sweep — VERIFIED AND COMPLETE

Protocol:

    src/experiments/refined/alpha_sweep_protocol.py
    docs/D046_ALPHA_SWEEP_PROTOCOL.md

Frozen grid:

    alpha = (0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 0.99)
    R = 300 per alpha
    total simulations = 7200
    bootstrap = 5000 matched full-replication blocks

Analysis/production:

    src/experiments/refined/alpha_sweep_analysis.py
    src/experiments/refined/alpha_sweep_production.py
    scripts/run_refined_alpha_sweep.py
    scripts/run_refined_alpha_sweep.slurm
    scripts/finalize_refined_alpha_sweep.py
    scripts/finalize_refined_alpha_sweep.slurm

Execution:

    array job 1526697: 48/48 tasks COMPLETED, exit 0:0
    checkpoints: 2400/2400
    non-empty stderr: 0
    finalizer 1527139: COMPLETED exit 0 on ruby047
    production/finalizer commit: 5283a338d75560f27d28a19d48e07f86cdeada07

The final analysis verified the exact alpha=0 economic-path topology null over
all 300 replications.

Main result mapping:

    docs/D046_RESULTS.md

Key result: absolute market activity declines with alpha while topology
differentiation strengthens into the coherent pre-boundary region around
alpha=0.85; alpha=0.95--0.99 shows high-social-weight ranking reversals.

Status: VERIFIED AND COMPLETE.

## D047 exploratory beta sweep protocol

    src/experiments/refined/beta_sweep_protocol.py
    docs/D047_BETA_SWEEP_PROTOCOL.md

Frozen exploratory design:

    experiment seed = 2026090701
    alpha anchor = 0.85
    beta = (0.00, 0.01, 0.10, 0.50, 1.00, 2.00, 5.00, 10.00, 100.00, 1000.00)
    paired replications per beta = 300
    total simulations = 9000
    bootstrap seed = 2026090702
    bootstrap draws = 5000
    confidence = 95%

The grid contains the exact beta=0 no-selectivity control, the D043 beta=1
anchor, the report's beta=2--5 transition region, and the report-scale high-beta
range through 1000.

D047 is exploratory because alpha=0.85 was selected after D046 outcome
inspection.

## D047 matched-block analysis

    src/experiments/refined/beta_sweep_analysis.py

New wrapper schema:

    BetaSweepTreatmentRecord(beta, treatment: ConfirmatoryTreatmentRecord)

This avoids mutating the frozen D045/D046 treatment record schema.

One replication is the complete matched block containing all ten beta values and
all R/SW/SF treatments. Final analysis validates that shock seeds,
initial-state seeds, and topology-specific graph seeds are identical across the
beta grid within each replication.

Outputs at every beta:

    topology means
    absolute/relative topology gaps where meaningful
    R-SW, R-SF, SW-SF contrasts
    matched-block percentile bootstrap intervals

No Holm/FWER family is attached to D047.

## D047 resumable production layer

    src/experiments/refined/beta_sweep_production.py

For each beta, the frozen D043 baseline is copied with only:

    alpha -> 0.85
    beta  -> selected beta grid value

changed. All remaining D043 parameters and D044 evaluation definitions stay
fixed.

Checkpoint path:

    results/refined/beta_sweep/checkpoints/beta_XX/replication_XXXX.json

Each checkpoint is one complete R/SW/SF triplet for one beta/replication pair.
Finalization requires all `10 x 300 = 3000` checkpoints.

Final planned artifacts:

    beta_sweep_records.csv
    beta_sweep_metadata.json
    beta_sweep_analysis.json
    beta_topology_means.csv
    beta_topology_gaps.csv
    beta_pairwise_contrasts.csv

## D047 execution layer

    scripts/run_refined_beta_sweep.py
    scripts/run_refined_beta_sweep.slurm
    scripts/finalize_refined_beta_sweep.py
    scripts/finalize_refined_beta_sweep.slurm

Array design:

    60 tasks total
    10 beta slices x 6 blocks
    50 paired replications per task
    max 16 concurrent tasks
    one CPU/task
    4 GB/task
    one-hour walltime/task

No Slurm partition/account is guessed. Login-node execution remains guarded.

## D047 tests

    tests/test_refined_beta_sweep_protocol.py
    tests/test_refined_beta_sweep_analysis.py
    tests/test_refined_beta_sweep_production.py
    tests/test_refined_beta_sweep_slurm.py

31 new D047 tests were added after the verified 689-test checkpoint.

Expected next checkpoint:

    720 passed

## Current gate

1. Pull latest `refined-model`.
2. Run all refined tests; expected `720 passed`.
3. Confirm clean working tree.
4. Do not submit D047 until the 720-test gate is green.
5. Before `sbatch`, create `results/refined/beta_sweep` so Slurm can open logs.
6. Do not inspect partial beta/topology curves; finalization requires all 3000 checkpoints.

Formal stability remains separate: equilibrium X*, full Jacobian J*, `spr(J*)`,
and Lyapunov analysis. The spectral radius of row-stochastic W is never the
market-stability criterion.
