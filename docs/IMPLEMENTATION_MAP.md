# Refined Model Implementation Map

Last updated: 2026-09-04

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

D045 protection:

    PairedReplicationPlan.n_agents
    PairedReplicationPlan.n_periods
    PairedReplicationPlan.parameters_fingerprint
    PairedReplicationPlan.validate_parameters(...)

The common shock plan is cryptographically bound to the exact refined
parameter vector used to generate it. Treatment construction fails on a later
parameter mismatch.

Status: VERIFIED at the 660-test checkpoint.

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

`rolling_action_covariance` uses rolling sums/squared sums and the exact Eq.
(240) identity rather than constructing an N-by-N covariance matrix at every
endpoint. A direct NumPy covariance regression test verifies algebraic
equivalence.

Status: VERIFIED at the 660-test checkpoint.

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

plus the remaining frozen information/noise parameters in the baseline document.

Status: VERIFIED.

## D042 / D044 frozen market evaluation

    src/experiments/refined/market_calibration.py
    src/experiments/refined/no_social_calibration_paths.py
    src/experiments/refined/market_calibration_run.py
    src/experiments/refined/frozen_market_calibration.py
    docs/FINAL_MARKET_CALIBRATION.md

Production calibration job 1505911 completed 500+500 no-social runs.

Frozen values:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

Status: VERIFIED.

## Paired confirmatory market runner — Phase 8

    src/experiments/refined/confirmatory_runner.py

Canonical API:

    ConfirmatoryTreatmentRecord
    ConfirmatorySmokeResult
    run_paired_confirmatory_replication(...)
    run_paired_confirmatory_smoke(...)
    write_paired_confirmatory_smoke(...)

Per treatment it records semantic seeds, economic-path fingerprint, market
outcomes, CID classification, graph diagnostics, realised-influence diagnostics,
and the production mechanism summaries:

    mean_pairwise_action_covariance
    mean_sum_individual_action_variances
    mean_aggregate_order_flow_variance

The same runner now also supplies D046 via `alpha_override`, without changing
any other frozen D043 parameter.

Status: VERIFIED at the 660-test checkpoint.

## Phase-8 smoke — VERIFIED

    scripts/run_refined_confirmatory_smoke.py
    scripts/run_refined_confirmatory_smoke.slurm
    tests/test_refined_confirmatory_runner.py

Smoke namespace `2026090401`, 2 paired replications, R/SW/SF,
baseline+alpha0. Slurm job 1509863 completed successfully. Both alpha0
replications produced exactly one non-attention economic-path fingerprint across
R/SW/SF.

## D045 frozen production protocol

    src/experiments/refined/confirmatory_protocol.py
    docs/D045_CONFIRMATORY_PROTOCOL.md

Frozen design:

    production seed = 2026090402
    n paired replications = 1000
    bootstrap seed = 2026090403
    bootstrap draws = 10000
    confidence = 95%
    familywise alpha = 0.05
    topology pairs = R-SW, R-SF, SW-SF

Primary family: six market/CID outcomes x three pairs = 18 hypotheses.
Mechanism family: four mechanism outcomes x three pairs = 12 hypotheses.
Holm FWER is applied separately to the two predeclared families. Secondary
outcomes are pointwise exploratory.

The first production run contains baseline `alpha=0.75` only. Alpha zero is not
repeated at scale because its exact topology-null property already passed the
Phase-8 end-to-end smoke.

## D045 matched-triplet inference

    src/experiments/refined/confirmatory_inference.py

Implements complete-triplet validation, topology means/gaps, all three named
pairwise contrasts, triplet-preserving 10000-draw bootstrap, percentile
intervals, centered bootstrap p-values, Holm FWER for the primary and mechanism
families, exploratory secondary intervals, and censoring-aware counts.

## D045 resumable production and execution

    src/experiments/refined/confirmatory_production.py
    scripts/run_refined_confirmatory_production.py
    scripts/run_refined_confirmatory_production.slurm
    scripts/finalize_refined_confirmatory_production.py
    scripts/finalize_refined_confirmatory_production.slurm

Per-replication checkpoint:

    results/refined/confirmatory_production/replications/replication_XXXX.json

Each checkpoint is one complete R/SW/SF triplet and stores a SHA-256
configuration fingerprint. Finalization is gated on all 1000 checkpoints.

Production array `1511972` completed all 10 tasks at exit code 0 on commit
`b5fbf52dd988637d90d7b5bc5c346c20551b66be`. Finalization job `1512116`
completed successfully on the same commit.

Final verified artifacts:

    confirmatory_records.csv
    confirmatory_metadata.json
    confirmatory_analysis.json
    topology_means.csv
    topology_gaps.csv
    pairwise_contrasts.csv

Canonical result summary:

    docs/D045_RESULTS.md

Status: VERIFIED AND FROZEN.

## D045 tests

    tests/test_refined_paired_parameter_binding.py
    tests/test_refined_action_covariance_vectorised.py
    tests/test_refined_confirmatory_protocol.py
    tests/test_refined_confirmatory_inference.py
    tests/test_refined_confirmatory_production.py
    tests/test_refined_confirmatory_production_slurm.py

Final D045 checkpoint:

    660 passed

## D046 exploratory alpha sweep protocol

    src/experiments/refined/alpha_sweep_protocol.py
    docs/D046_ALPHA_SWEEP_PROTOCOL.md

Frozen exploratory design:

    experiment seed = 2026090404
    alpha grid = (0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 1.00)
    paired replications per alpha = 300
    total simulations = 7200
    bootstrap seed = 2026090405
    bootstrap draws = 5000
    confidence = 95%

D046 is OAT diagnostic/regime mapping, not a second confirmatory family.
D043/D044 values remain fixed. Alpha zero and alpha one are explicit endpoints,
and alpha 0.75 retains the D045 anchor.

## D046 matched-block analysis

    src/experiments/refined/alpha_sweep_analysis.py

One replication is treated as a complete matched block containing all eight
alpha values and all three topology treatments. Bootstrap resampling preserves
that full block. Independent resampling by alpha or topology is prohibited.

Outputs include, at every alpha:

    topology means
    absolute/relative topology gaps
    R-SW, R-SF, SW-SF contrasts
    percentile bootstrap intervals

No Holm/FWER rejection family is attached to D046. The final analysis also
requires the exact alpha=0 economic-path topology-null property for every
replication.

## D046 resumable production layer

    src/experiments/refined/alpha_sweep_production.py

Checkpoint path:

    results/refined/alpha_sweep/checkpoints/alpha_XX/replication_XXXX.json

Each checkpoint is one complete R/SW/SF triplet for one alpha/replication pair.
Finalization requires all `8 x 300 = 2400` checkpoints.

Final planned artifacts:

    alpha_sweep_records.csv
    alpha_sweep_metadata.json
    alpha_sweep_analysis.json
    alpha_topology_means.csv
    alpha_topology_gaps.csv
    alpha_pairwise_contrasts.csv

## D046 execution layer

    scripts/run_refined_alpha_sweep.py
    scripts/run_refined_alpha_sweep.slurm
    scripts/finalize_refined_alpha_sweep.py
    scripts/finalize_refined_alpha_sweep.slurm

Array design:

    48 tasks total
    8 alpha slices x 6 blocks
    50 paired replications per task
    max 16 concurrent tasks
    one CPU/task
    4 GB/task
    one-hour walltime/task

No partition/account is guessed. Login-node execution remains guarded.

## D046 tests

    tests/test_refined_alpha_sweep_protocol.py
    tests/test_refined_alpha_sweep_analysis.py
    tests/test_refined_alpha_sweep_production.py
    tests/test_refined_alpha_sweep_slurm.py

28 new test cases are added beyond the verified 660 checkpoint.

Expected next checkpoint:

    688 passed

## Current gate

1. Pull latest `refined-model`.
2. Run all refined tests; expected `688 passed`.
3. Confirm clean working tree.
4. Do not submit D046 until the 688-test gate is green.
5. Before `sbatch`, create `results/refined/alpha_sweep` so Slurm can open log files.
6. Do not inspect partial alpha/topology curves; finalization requires all 2400 checkpoints.

Formal stability remains separate: equilibrium X*, full Jacobian J*, `spr(J*)`,
and Lyapunov analysis. The spectral radius of row-stochastic W is never the
market-stability criterion.
