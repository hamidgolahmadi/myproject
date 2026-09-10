# Refined Model Implementation Map

Last updated: 2026-09-10

Scientific source of truth: `report1_25_08_2026.pdf`. Legacy code is reference only.

## Core runtime

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

    W_{t-1} -> theta_t,v_t,s_t -> b_t -> vhat_t -> m_t
    -> desired action -> executed action -> x_t -> F_t -> p_t
    -> r_t -> pi_t -> R_t -> z_t -> W_t

Status: VERIFIED.

## Paired design and semantic randomness

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py
    src/experiments/refined/treatments.py

Common-random-number sweeps preserve shock path, neutral non-network initial
state, and topology-specific graph seeds across parameter values whenever the
swept parameter does not alter those distributions.

Status: VERIFIED.

## Structural, market, CID, and mechanism diagnostics

    src/topologies/refined/generators.py
    src/topologies/refined/diagnostics.py
    src/experiments/refined/structural.py
    src/experiments/refined/market_metrics.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/cid.py
    src/experiments/refined/cid_events.py
    src/experiments/refined/influence_metrics.py

D048 adds a descriptive sidecar only:

    src/experiments/refined/reputation_diagnostics.py

It reports raw local reputation dispersion before the Eq. (58) sigma_0 floor
and the ratio of raw dispersion to sigma_0. It does not change the economic
transition or any D045--D047 outcome definition.

## Frozen D043/D044 baseline and evaluation

    src/experiments/refined/baseline_specification.py
    src/experiments/refined/frozen_market_calibration.py
    docs/REFINED_BASELINE.md
    docs/FINAL_MARKET_CALIBRATION.md

    N=100, K=6, T=1000, q=5, p_sw=0.02, a0=1.0
    alpha=0.75, gamma_R=0.9, beta=1.0, sigma_0=0.0005
    c_ret=0.0030364359162156455
    c_bel=0.004182211355781272
    c_F=0.11381404220614316
    c_CID=1.8326578831721285

Status: VERIFIED AND FROZEN.

## Common treatment engine

    src/experiments/refined/confirmatory_runner.py

The frozen `ConfirmatoryTreatmentRecord` schema remains unchanged. D047 wraps
it in `BetaSweepTreatmentRecord`; D048 wraps it in `GammaSweepTreatmentRecord`.

## D045 confirmatory production — COMPLETE

    src/experiments/refined/confirmatory_protocol.py
    src/experiments/refined/confirmatory_inference.py
    src/experiments/refined/confirmatory_production.py
    docs/D045_CONFIRMATORY_PROTOCOL.md
    docs/D045_RESULTS.md

Array 1511972 and finalizer 1512116 completed on commit
`b5fbf52dd988637d90d7b5bc5c346c20551b66be`.

## D046 alpha sweep — COMPLETE

    src/experiments/refined/alpha_sweep_protocol.py
    src/experiments/refined/alpha_sweep_analysis.py
    src/experiments/refined/alpha_sweep_production.py
    docs/D046_ALPHA_SWEEP_PROTOCOL.md
    docs/D046_RESULTS.md

    alpha=(0,.2,.4,.6,.75,.85,.95,.99)
    300 paired replications per alpha
    7200 simulations

Array 1526697 and finalizer 1527139 completed on commit
`5283a338d75560f27d28a19d48e07f86cdeada07`.

## D047 beta sweep — COMPLETE

    src/experiments/refined/beta_sweep_protocol.py
    src/experiments/refined/beta_sweep_analysis.py
    src/experiments/refined/beta_sweep_production.py
    src/experiments/refined/beta_sweep_plotting.py
    docs/D047_BETA_SWEEP_PROTOCOL.md
    docs/D047_RESULTS.md

    alpha=.85
    beta=(0,.01,.1,.5,1,2,5,10,100,1000)
    300 paired replications per beta
    9000 simulations

Array 1536715 completed 60/60 tasks with 3000/3000 checkpoints and no non-empty
stderr. Finalizer 1537380 completed on the same commit
`6c94d6b92014185c4ed4799d7846b284b31f8a27`.

D047 plotting is closed. The preferred thesis display is symlog; the default
plot command generates five main figure types in PNG and vector PDF.

## D048 gamma_R sweep — IMPLEMENTED / TEST PENDING

Protocol:

    docs/D048_GAMMA_SWEEP_PROTOCOL.md
    src/experiments/refined/gamma_sweep_protocol.py

Frozen design:

    experiment seed=2026091001
    bootstrap seed=2026091002
    alpha=.85
    beta=5.0
    gamma_R=(0,.5,.8,.9,.95,.98,.99,.995,.999)
    300 paired replications per gamma_R
    2700 matched gamma/replication blocks
    8100 simulations
    5000 complete-block bootstrap draws

Implementation:

    src/experiments/refined/reputation_diagnostics.py
    src/experiments/refined/gamma_sweep_analysis.py
    src/experiments/refined/gamma_sweep_runner.py
    src/experiments/refined/gamma_sweep_production.py
    scripts/run_refined_gamma_sweep.py
    scripts/run_refined_gamma_sweep.slurm
    scripts/finalize_refined_gamma_sweep.py
    scripts/finalize_refined_gamma_sweep.slurm

Wrapper schema:

    GammaSweepTreatmentRecord(
        gamma_R,
        mean_raw_local_reputation_std,
        mean_raw_local_reputation_std_over_sigma0,
        treatment: ConfirmatoryTreatmentRecord,
    )

Checkpoint path:

    results/refined/gamma_sweep/checkpoints/gamma_XX/replication_XXXX.json

Production array design:

    54 tasks = 9 gamma slices x 6 blocks
    50 paired replications/task
    max 16 concurrent
    1 CPU/task, 4 GB/task, 1 hour/task

Finalization requires all 2700 checkpoints. One bootstrap unit is the complete
9-gamma x 3-topology replication block. D048 is exploratory OAT; no new
Holm/FWER family is declared.

D048 test files:

    tests/test_refined_gamma_sweep_protocol.py
    tests/test_refined_reputation_diagnostics.py
    tests/test_refined_gamma_sweep_runner.py
    tests/test_refined_gamma_sweep_analysis.py
    tests/test_refined_gamma_sweep_production.py
    tests/test_refined_gamma_sweep_slurm.py

There are 34 new D048 tests. Together with the three unshown final-plotting tests
added after the last verified 729-test checkpoint, the expected next full gate
is 766. This count is not verified until Iridis reports it.

## Next gate

1. Pull latest `refined-model`.
2. Run all refined tests and confirm a clean working tree.
3. Run a tiny compute-node smoke at gamma_R={0,.9,.999}.
4. Submit the 54-task production array only after the smoke passes.
5. Do not inspect partial D048 outcome curves.
6. Finalize only after 2700/2700 checkpoints and clean one-commit provenance.

Formal stability remains separate: equilibrium X*, full Jacobian J*, `spr(J*)`,
and Lyapunov analysis. The spectral radius of row-stochastic W is not the market
stability criterion.
