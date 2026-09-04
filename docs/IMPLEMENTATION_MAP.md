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

Binding transition:

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

Status: VERIFIED, including generated-treatment alpha=0 topology-null controls.

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

Status: VERIFIED.

## Realised influence — Eqs. (251)-(265)

    src/experiments/refined/influence_metrics.py

Implements:

    normalised attention entropy
    effective source count
    source influence shares
    influence HHI
    structural-hub realised influence
    attention overlap
    attention mobility

Structural hubs come from directed in-degree in G, never W_t.

Status: VERIFIED.

Eqs. (266)-(267) KL-to-transition-prior remain deferred with attention inertia.

## D043 frozen baseline

    src/experiments/refined/baseline_specification.py
    docs/REFINED_BASELINE.md

Canonical API:

    RefinedBaselineSpecification
    first_refined_baseline_specification()

Frozen design:

    N=100, K=6, T=1000, q=5, p_sw=0.02, a0=1.0

Frozen parameter vector includes:

    alpha=0.75
    kappa=2.4
    chi=0.02
    lambda_price=0.0002
    gamma_R=0.9
    beta=1.0
    sigma_0=0.0005

plus the remaining D043 information/noise parameters recorded in
`docs/REFINED_BASELINE.md`.

Status: VERIFIED.

## D042 calibration method and production runner

Method:

    src/experiments/refined/market_calibration.py

No-social shared path:

    src/experiments/refined/no_social_calibration_paths.py

Production runner:

    src/experiments/refined/market_calibration_run.py
    scripts/run_refined_market_calibration.py
    scripts/run_refined_market_calibration.slurm

Production protocol:

    T=1000, B=0, L=50
    500 scale runs, seed 2026090201
    500 threshold runs, seed 2026090202
    equal CID weights
    c_CID = 95th percentile of run-level peak CID, method=higher
    L_stab=50

Slurm job 1505911 completed 500+500 with exit code 0.

Status: VERIFIED.

## D044 frozen numerical calibration

    src/experiments/refined/frozen_market_calibration.py
    docs/FINAL_MARKET_CALIBRATION.md

Canonical API:

    frozen_reference_scales()
    first_frozen_market_evaluation_calibration()

Frozen values:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

Fingerprints:

    configuration = 9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202
    scales        = 1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476

Status: VERIFIED.

## Paired confirmatory market runner — Phase 8

Core:

    src/experiments/refined/confirmatory_runner.py

API:

    ConfirmatoryTreatmentRecord
    ConfirmatorySmokeResult
    run_paired_confirmatory_replication(...)
    run_paired_confirmatory_smoke(...)
    write_paired_confirmatory_smoke(...)

Smoke CLI / batch wrapper:

    scripts/run_refined_confirmatory_smoke.py
    scripts/run_refined_confirmatory_smoke.slurm

Tests:

    tests/test_refined_confirmatory_runner.py

Per treatment the runner persists:

    semantic experiment / replication / graph / shock / initial-state identifiers
    economic-path SHA-256 fingerprint excluding W
    run-level market outcomes
    CID peak / exceedance / duration / stabilisation / censoring
    structural graph diagnostics
    time-averaged realised-influence diagnostics

The runner always uses the common-random-number paired treatment constructor.
It does not estimate treatment effects or rank topologies.

Status: VERIFIED at the 619-test checkpoint.

## Confirmatory smoke negative control — VERIFIED

Smoke-only seed namespace:

    2026090401

Design:

    2 paired replications
    R / SW / SF
    baseline alpha=0.75
    alpha0_control alpha=0
    12 simulations total

Iridis Slurm job:

    job = 1509863
    host = ruby047
    state = COMPLETED
    exit code = 0
    elapsed = 00:01:50
    CPU efficiency = 99.09%
    memory utilised = 364.41 MB

Within every alpha0 replication, R/SW/SF produced exactly the same complete
non-attention economic-path fingerprint:

    replication 0 -> 1 unique fingerprint
    replication 1 -> 1 unique fingerprint

This is the required end-to-end topology-null check. G and W remain distinct
across treatments, but market paths do not depend on them when alpha=0.

Persistence:

    results/refined/confirmatory_smoke/confirmatory_smoke_records.csv
    results/refined/confirmatory_smoke/confirmatory_smoke_metadata.json

The smoke is explicitly non-final and must not be used to rank R/SW/SF.

## Current gate

1. The paired confirmatory runner and its end-to-end smoke are verified.
2. Do not launch a large confirmatory topology Monte Carlo yet.
3. Next freeze a D045 production confirmatory protocol before inspecting production outcomes:
   - disjoint production seed namespace;
   - paired replication count;
   - exact baseline/control regime set for the first production run;
   - resumable checkpoint/artifact layout;
   - pre-specified treatment contrasts and summary statistics.
4. Implement the resumable production layer against D045.
5. Test it deterministically and run a small production-layer smoke before the large Iridis job.

Formal stability remains separate: equilibrium X*, full Jacobian J*, `spr(J*)`,
and Lyapunov analysis. The spectral radius of row-stochastic W is never the
market-stability criterion.
