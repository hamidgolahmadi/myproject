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
    src/experiments/refined/reputation_diagnostics.py

The reputation sidecar reports raw local reputation dispersion before the
Eq. (58) sigma_0 floor and its ratio to sigma_0. It is diagnostic only and does
not alter the economic transition.

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

The frozen `ConfirmatoryTreatmentRecord` schema remains unchanged. Later
experiments wrap it rather than mutating the D045 schema:

    D047 -> BetaSweepTreatmentRecord
    D048 -> GammaSweepTreatmentRecord
    D049 -> JointTreatmentRecord

## D045 confirmatory production — COMPLETE

    docs/D045_CONFIRMATORY_PROTOCOL.md
    docs/D045_RESULTS.md

Array 1511972 and finalizer 1512116 completed on commit
`b5fbf52dd988637d90d7b5bc5c346c20551b66be`.

## D046 alpha sweep — COMPLETE

    docs/D046_ALPHA_SWEEP_PROTOCOL.md
    docs/D046_RESULTS.md

    alpha=(0,.2,.4,.6,.75,.85,.95,.99)
    300 paired replications per alpha
    7200 simulations

Array 1526697 and finalizer 1527139 completed on commit
`5283a338d75560f27d28a19d48e07f86cdeada07`.

## D047 beta sweep — COMPLETE

    docs/D047_BETA_SWEEP_PROTOCOL.md
    docs/D047_RESULTS.md

    alpha=.85
    beta=(0,.01,.1,.5,1,2,5,10,100,1000)
    300 paired replications per beta
    9000 simulations

Array 1536715 completed 60/60 tasks with 3000/3000 checkpoints and no non-empty
stderr. Finalizer 1537380 completed on commit
`6c94d6b92014185c4ed4799d7846b284b31f8a27`.

D047 plotting is closed. Preferred thesis display is symlog.

## D048 gamma_R sweep — COMPLETE AND FINALIZED

    docs/D048_GAMMA_SWEEP_PROTOCOL.md
    docs/D048_RESULTS.md
    src/experiments/refined/gamma_sweep_protocol.py
    src/experiments/refined/gamma_sweep_analysis.py
    src/experiments/refined/gamma_sweep_runner.py
    src/experiments/refined/gamma_sweep_production.py

Frozen design:

    experiment seed=2026091001
    bootstrap seed=2026091002
    alpha=.85
    beta=5.0
    gamma_R=(0,.5,.8,.9,.95,.98,.99,.995,.999)
    300 paired replications per gamma_R
    8100 simulations

Execution:

    refined tests = 766 passed
    smoke job 1541812 = 3/3 COMPLETED 0:0
    production array 1542666 = 54/54 COMPLETED 0:0
    checkpoints = 2700/2700
    non-empty stderr = 0
    finalizer 1544241 = COMPLETED 0:0
    production/finalizer commit = ce3630faf86d7f55b3535d2a8b902d5c677a6f0c

Main result: persistence amplifies topology-dependent common attention and
correlated order flow over a broad interior range, but at gamma_R=.999 raw
reputation dispersion falls below the fixed sigma_0 scale and the mechanism
partially reverses. Threshold-exceeding remains zero throughout D048.

Status: VERIFIED, COMPLETE, FINALIZED, AND DOCUMENTED.

## D049 joint alpha-beta-gamma_R interaction experiment — IMPLEMENTED / TEST PENDING

Canonical protocol:

    docs/D049_JOINT_INTERACTION_PROTOCOL.md
    src/experiments/refined/joint_interaction_protocol.py

Reduced factorial:

    alpha=(.40,.85,.99)
    beta=(0,1,5,100)
    gamma_R=(0,.90,.99,.999)

This gives 48 factorial cells, plus:

    C_ALPHA0 = (alpha=0, beta=5, gamma_R=.90)
    A_D043   = (alpha=.75, beta=1, gamma_R=.90)

Total:

    50 cells
    300 paired replications/cell
    15000 resumable cell/replication checkpoints
    45000 simulations
    bootstrap=5000 complete 50-cell x 3-topology replication blocks
    experiment seed=2026091003
    bootstrap seed=2026091004

D049 keeps sigma_0 fixed at .0005. It is sequential exploratory interaction
mapping because factor levels were selected after D046--D048 outcome inspection.
No new Holm/FWER family is declared.

Main signed interaction topology estimand:

    D(alpha,beta,gamma_R;Y) = Y_SF - Y_SW

The protocol freezes six genuine interaction contrasts plus three nonlinear
boundary-shift contrasts. The priority outcome chain is:

    attention overlap
      -> pairwise action covariance
      -> aggregate order-flow variance
      -> mean absolute order flow
      -> return volatility
      -> peak CID

Implementation:

    src/experiments/refined/joint_interaction_protocol.py
    src/experiments/refined/joint_interaction_analysis.py
    src/experiments/refined/joint_interaction_runner.py
    src/experiments/refined/joint_interaction_production.py
    scripts/run_refined_joint_interaction.py
    scripts/run_refined_joint_interaction.slurm
    scripts/finalize_refined_joint_interaction.py
    scripts/finalize_refined_joint_interaction.slurm

Checkpoint path:

    results/refined/joint_interaction/checkpoints/cell_XX/replication_XXXX.json

Production array design:

    300 tasks = 50 cells x 6 blocks
    50 paired replications/task
    max 16 concurrent
    1 CPU/task, 4 GB/task, 1 hour/task

Finalizer requires all 15000 checkpoints and verifies both cross-cell CRN and
the exact alpha=0 economic-path topology null before writing final artifacts.

D049 test files:

    tests/test_refined_joint_interaction_protocol.py
    tests/test_refined_joint_interaction_runner.py
    tests/test_refined_joint_interaction_analysis.py
    tests/test_refined_joint_interaction_production.py
    tests/test_refined_joint_interaction_slurm.py

Exactly 34 D049 test cases were added after the verified 766-test checkpoint.
Expected next full refined gate, if all additions pass:

    800 passed

Do not call this verified until Iridis reports it.

## Next gate

1. Pull latest `refined-model`.
2. Run `python -m pytest -q tests/test_refined_*.py`.
3. Confirm exactly 800 passed and a clean working tree before D049 execution.
4. Run a compute-node smoke at cell indexes 48, 49, 26, and 47:
   alpha-zero control, D043 anchor, interior (.85,5,.99), and boundary
   (.99,100,.999).
5. Only after that smoke passes, create a fresh production output directory and
   submit the 300-task array.
6. Do not inspect partial D049 surfaces.
7. Finalize only after 15000/15000 checkpoints and one-commit provenance.

After D049 interpretation, proceed to heterogeneity.

Formal stability remains separate: equilibrium X*, full Jacobian J*, `spr(J*)`,
and Lyapunov analysis. The spectral radius of row-stochastic W is not the market
stability criterion.
