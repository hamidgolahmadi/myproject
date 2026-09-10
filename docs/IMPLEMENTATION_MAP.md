# Refined Model Implementation Map

Last updated: 2026-09-10

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

The same semantic seed construction is reused across OAT values when the swept
parameter does not change the shock or neutral-initial-state distribution.

Status: VERIFIED.

## Topologies and structural validation — D041

    src/topologies/refined/generators.py
    src/topologies/refined/diagnostics.py
    src/experiments/refined/structural.py
    src/experiments/refined/calibration.py
    src/experiments/refined/structural_io.py

Matched design:

    N=100, K=6, q=5, p_sw=0.02, a0=1.0

1000 graphs per R/SW/SF topology passed the structural gate.

Status: VERIFIED.

## Market/CID/mechanism diagnostics

    src/experiments/refined/market_metrics.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/cid.py
    src/experiments/refined/cid_events.py
    src/experiments/refined/influence_metrics.py

Implements return/mispricing/order-flow outcomes, rolling CID, threshold and
stabilisation summaries, action-covariance decomposition, attention entropy,
effective sources, realised influence HHI, structural-hub realised influence,
attention overlap, and attention mobility.

D048 adds a descriptive sidecar only:

    src/experiments/refined/reputation_diagnostics.py

It computes the raw local reputation standard deviation before the Eq. (58)
`sigma_0` floor and the ratio of that raw dispersion to `sigma_0`. This does not
modify the economic transition or any D045--D047 outcome definition.

Status: VERIFIED through D047; D048 sidecar awaits the next full test gate.

## D043 frozen baseline

    src/experiments/refined/baseline_specification.py
    docs/REFINED_BASELINE.md

    N=100, K=6, T=1000, q=5, p_sw=0.02, a0=1.0
    alpha=0.75, kappa=2.4, chi=0.02, lambda_price=0.0002
    gamma_R=0.9, beta=1.0, sigma_0=0.0005

Status: VERIFIED AND FROZEN.

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

Status: VERIFIED AND FROZEN.

## Common fixed-topology treatment engine — Phase 8

    src/experiments/refined/confirmatory_runner.py

Canonical treatment schema remains:

    ConfirmatoryTreatmentRecord

D047 wraps it in `BetaSweepTreatmentRecord`; D048 wraps it in
`GammaSweepTreatmentRecord`. The frozen D045 record schema is not mutated.

Status: VERIFIED.

## D045 confirmatory production — COMPLETE

    src/experiments/refined/confirmatory_protocol.py
    src/experiments/refined/confirmatory_inference.py
    src/experiments/refined/confirmatory_production.py
    docs/D045_CONFIRMATORY_PROTOCOL.md
    docs/D045_RESULTS.md

Production array 1511972 and finalizer 1512116 completed on commit
`b5fbf52dd988637d90d7b5bc5c346c20551b66be`.

Status: VERIFIED AND FROZEN.

## D046 exploratory alpha sweep — COMPLETE

    src/experiments/refined/alpha_sweep_protocol.py
    src/experiments/refined/alpha_sweep_analysis.py
    src/experiments/refined/alpha_sweep_production.py
    scripts/run_refined_alpha_sweep.py
    scripts/run_refined_alpha_sweep.slurm
    scripts/finalize_refined_alpha_sweep.py
    scripts/finalize_refined_alpha_sweep.slurm
    docs/D046_ALPHA_SWEEP_PROTOCOL.md
    docs/D046_RESULTS.md

Frozen design:

    alpha=(0,.2,.4,.6,.75,.85,.95,.99)
    R=300 per alpha
    7200 simulations
    5000 complete-block bootstrap draws

Production 1526697 and finalizer 1527139 completed on commit
`5283a338d75560f27d28a19d48e07f86cdeada07`.

Status: VERIFIED AND COMPLETE.

## D047 exploratory beta sweep — COMPLETE

    src/experiments/refined/beta_sweep_protocol.py
    src/experiments/refined/beta_sweep_analysis.py
    src/experiments/refined/beta_sweep_production.py
    scripts/run_refined_beta_sweep.py
    scripts/run_refined_beta_sweep.slurm
    scripts/finalize_refined_beta_sweep.py
    scripts/finalize_refined_beta_sweep.slurm
    docs/D047_BETA_SWEEP_PROTOCOL.md
    docs/D047_RESULTS.md

Frozen design:

    alpha=.85
    beta=(0,.01,.1,.5,1,2,5,10,100,1000)
    R=300 per beta
    9000 simulations
    5000 complete 10-beta x 3-topology bootstrap blocks

Production array 1536715 completed 60/60 tasks, 3000/3000 checkpoints, no
non-empty stderr; finalizer 1537380 completed on the same commit
`6c94d6b92014185c4ed4799d7846b284b31f8a27`.

Plotting layer:

    src/experiments/refined/beta_sweep_plotting.py
    scripts/plot_refined_beta_sweep.py

Preferred thesis view is symlog, retaining beta=0 while resolving the broad
positive beta range. Default generation now produces five main figure types in
PNG and vector PDF.

Status: COMPLETE. Plotting is closed unless a report-formatting need arises.

## D048 exploratory gamma_R sweep — IMPLEMENTED / TEST PENDING

Scientific protocol:

    docs/D048_GAMMA_SWEEP_PROTOCOL.md
    src/experiments/refined/gamma_sweep_protocol.py

Frozen design:

    alpha anchor=.85
    beta anchor=5.0
    gamma_R=(0,.5,.8,.9,.95,.98,.99,.995,.999)
    R=300 per gamma_R
    2700 matched gamma/replication blocks
    8100 simulations
    bootstrap seed=2026091002
    bootstrap draws=5000

The anchor beta=5 is post-D047 and explicitly exploratory: it is an interior
amplification point rather than the beta=100--1000 saturation range.

Analysis and wrapper schema:

    src/experiments/refined/gamma_sweep_analysis.py

    GammaSweepTreatmentRecord(
        gamma_R,
        mean_raw_local_reputation_std,
        mean_raw_local_reputation_std_over_sigma0,
        treatment: ConfirmatoryTreatmentRecord,
    )

Runner:

    src/experiments/refined/gamma_sweep_runner.py

This deliberately leaves `ConfirmatoryTreatmentRecord` unchanged while computing
the same frozen market/CID/mechanism record plus the two D048 reputation-scale
sidecars.

Production:

    src/experiments/refined/gamma_sweep_production.py
    scripts/run_refined_gamma_sweep.py
    scripts/run_refined_gamma_sweep.slurm
    scripts/finalize_refined_gamma_sweep.py
    scripts/finalize_refined_gamma_sweep.slurm

Checkpoint path:

    results/refined/gamma_sweep/checkpoints/gamma_XX/replication_XXXX.json

Array design:

    54 tasks total
    9 gamma slices x 6 blocks
    50 paired replications per task
    maximum 16 concurrent
    one CPU/task, 4 GB/task, one-hour walltime

Finalization requires all 2700 checkpoints and writes:

    gamma_sweep_records.csv
    gamma_sweep_metadata.json
    gamma_sweep_analysis.json
    gamma_topology_means.csv
    gamma_topology_gaps.csv
    gamma_pairwise_contrasts.csv

One bootstrap unit is the complete 9-gamma x 3-topology replication block.
No Holm/FWER family is attached; D048 remains exploratory OAT regime mapping.

Tests added for D048:

    tests/test_refined_gamma_sweep_protocol.py
    tests/test_refined_reputation_diagnostics.py
    tests/test_refined_gamma_sweep_runner.py
    tests/test_refined_gamma_sweep_analysis.py
    tests/test_refined_gamma_sweep_production.py
    tests/test_refined_gamma_sweep_slurm.py

Expected next full refined test gate: 765, assuming the unshown 732 plotting gate
and all 33 new D048 tests pass. Verify on Iridis; do not treat this count as
established until pytest reports it.

## Next execution gate

1. Pull latest `refined-model`.
2. Run all refined tests and confirm clean working tree.
3. Run a tiny compute-node D048 smoke at gamma_R={0,.9,.999}.
4. Only after smoke success submit the 54-task production array.
5. Do not inspect partial gamma/topology outcomes.
6. Finalize only after 2700/2700 checkpoints and clean provenance verification.

Formal stability remains separate: equilibrium X*, full Jacobian J*, `spr(J*)`,
and Lyapunov analysis. The spectral radius of row-stochastic W is never the
market-stability criterion.
