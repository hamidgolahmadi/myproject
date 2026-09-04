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

New D045 protection:

    PairedReplicationPlan.n_agents
    PairedReplicationPlan.n_periods
    PairedReplicationPlan.parameters_fingerprint
    PairedReplicationPlan.validate_parameters(...)

The common shock plan is now cryptographically bound to the exact refined
parameter vector used to generate it. Treatment construction fails on a later
parameter mismatch.

Status: implementation added after the 619-test checkpoint; awaiting 660-test gate.

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

D045 optimization:

`rolling_action_covariance` now uses rolling sums/squared sums and the exact
Eq. (240) identity rather than constructing an N-by-N covariance matrix at
every endpoint. A direct NumPy covariance regression test verifies algebraic
equivalence.

Status: VERIFIED through 619 before optimization; optimized form awaits 660-test gate.

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
and now the production mechanism summaries:

    mean_pairwise_action_covariance
    mean_sum_individual_action_variances
    mean_aggregate_order_flow_variance

Status: base runner VERIFIED at 619; added action-covariance fields await 660 gate.

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

Implements:

- complete-triplet validation;
- topology means;
- absolute and relative topology gaps;
- all three named pairwise contrasts;
- triplet-preserving bootstrap with 10000 draws;
- 95% percentile intervals;
- centered two-sided bootstrap p-values;
- Holm FWER for primary and mechanism families;
- pointwise exploratory secondary intervals;
- right-censored/non-stabilised counts without dropping censored runs.

Bootstrap is vectorised using multinomial replication counts in batches.

## D045 resumable production layer

    src/experiments/refined/confirmatory_production.py

Per-replication checkpoint:

    results/refined/confirmatory_production/replications/replication_XXXX.json

Each checkpoint is one complete R/SW/SF triplet and stores a SHA-256
configuration fingerprint. Resume accepts only exact matches.

Finalization is gated on all 1000 checkpoint files. Only then are written:

    confirmatory_records.csv
    confirmatory_metadata.json
    confirmatory_analysis.json
    topology_means.csv
    topology_gaps.csv
    pairwise_contrasts.csv

This prevents partial-sample topology inference from becoming a final artifact.

## D045 execution layer

Range CLI:

    scripts/run_refined_confirmatory_production.py

Array wrapper:

    scripts/run_refined_confirmatory_production.slurm

Design:

    array 0-9
    100 paired replications per task
    one CPU/task
    4 GB/task
    two-hour walltime/task

Finalization CLI / wrapper:

    scripts/finalize_refined_confirmatory_production.py
    scripts/finalize_refined_confirmatory_production.slurm

Both CLIs refuse Iridis login-node execution unless explicitly overridden.
No Slurm partition or account is guessed.

## D045 tests

    tests/test_refined_paired_parameter_binding.py
    tests/test_refined_action_covariance_vectorised.py
    tests/test_refined_confirmatory_protocol.py
    tests/test_refined_confirmatory_inference.py
    tests/test_refined_confirmatory_production.py
    tests/test_refined_confirmatory_production_slurm.py

41 new cases are expected beyond the verified 619 checkpoint.

Expected next checkpoint:

    660 passed

## Current gate

1. Pull latest `refined-model`.
2. Verify all refined tests: expected `660 passed`.
3. Verify clean working tree.
4. Only then submit the D045 10-task production array.
5. Do not finalize or interpret topology contrasts until all 1000 triplet checkpoints exist.

Formal stability remains separate: equilibrium X*, full Jacobian J*, `spr(J*)`,
and Lyapunov analysis. The spectral radius of row-stochastic W is never the
market-stability criterion.
