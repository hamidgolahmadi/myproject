# Project State

Last updated: 2026-09-10

## Project identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Branch:

    refined-model

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility only and never overrides the report.

Iridis setup:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model

## Latest verified code checkpoint

Latest explicitly verified full refined test gate in chat:

    766 passed in 32.42s
    working tree clean
    verified HEAD before D048 production = ce3630faf86d7f55b3535d2a8b902d5c677a6f0c

D049 has been implemented after this checkpoint. Exactly 34 new D049 test cases
have been added. Expected next full refined test count, if all additions pass:

    800 passed

Do not call 800 verified until Iridis reports it.

## Frozen structural and market baseline

D041 topology design:

    N=100, K=6, q=5, p_sw=0.02, a0=1.0

D043 homogeneous market baseline:

    T=1000
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

W0 is topology-specific uniform graph-supported attention.

D042/D044 frozen market evaluation:

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

    configuration = 9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202
    scales        = 1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476

These remain immutable through D049.

## D045 confirmatory fixed-topology experiment — COMPLETE

Production:

    alpha=0.75
    paired replications=1000
    simulations=3000
    production array=1511972
    finalizer=1512116
    commit=b5fbf52dd988637d90d7b5bc5c346c20551b66be

Main result: topology strongly reorganises the social transmission mechanism,
while realised price-instability effects are statistically detectable but
modest under the frozen baseline. Canonical results: `docs/D045_RESULTS.md`.

## D046 exploratory alpha sweep — COMPLETE

    alpha=(0,.2,.4,.6,.75,.85,.95,.99)
    300 paired replications per alpha
    7200 simulations
    bootstrap=5000 complete blocks
    array=1526697
    checkpoints=2400/2400
    finalizer=1527139
    commit=5283a338d75560f27d28a19d48e07f86cdeada07

Main result: absolute activity/volatility generally falls with alpha while
cross-topology differentiation strengthens into the coherent pre-boundary
region; alpha=.95--.99 displays a qualitatively different boundary regime.
Canonical results: `docs/D046_RESULTS.md`.

## D047 exploratory beta sweep — COMPLETE

    alpha anchor=.85
    beta=(0,.01,.1,.5,1,2,5,10,100,1000)
    300 paired replications per beta
    9000 simulations
    bootstrap=5000 complete 10-beta x 3-topology blocks
    production array=1536715
    checkpoints=3000/3000
    finalizer=1537380
    production/finalizer commit=6c94d6b92014185c4ed4799d7846b284b31f8a27

Main result: beta does not activate topology from zero; topology is already
present at beta=0 because alpha=.85 and G remains different. Stronger
selectivity mainly amplifies common attention, action covariance, and aggregate
order-flow variance, with several market gaps approaching a high-beta plateau.
Threshold-exceeding is zero throughout D047. Canonical results:
`docs/D047_RESULTS.md`.

D047 thesis plotting is closed.

## D048 exploratory gamma_R sweep — COMPLETE, FINALIZED, INTERPRETED

Canonical files:

    docs/D048_GAMMA_SWEEP_PROTOCOL.md
    docs/D048_RESULTS.md
    src/experiments/refined/gamma_sweep_protocol.py
    src/experiments/refined/gamma_sweep_analysis.py
    src/experiments/refined/gamma_sweep_runner.py
    src/experiments/refined/gamma_sweep_production.py

Frozen design:

    experiment seed=2026091001
    bootstrap seed=2026091002
    alpha anchor=.85
    beta anchor=5.0
    gamma_R=(0,.5,.8,.9,.95,.98,.99,.995,.999)
    paired replications per gamma=300
    simulations=8100
    bootstrap draws=5000

Execution:

    full refined test gate=766 passed in 32.42s
    smoke job=1541812; 3/3 COMPLETED 0:0
    production array=1542666; 54/54 COMPLETED 0:0
    checkpoints=2700/2700
    non-empty stderr=0
    finalizer=1544241 COMPLETED 0:0 on ruby047
    production/finalizer commit=ce3630faf86d7f55b3535d2a8b902d5c677a6f0c

Scientific result:

    gamma_R=0--.5      fast/noisy updating; high attention turnover
    gamma_R=.8--.95    persistent ranking; growing topology amplification
    gamma_R=.98--.995  strongest concentration/covariance differentiation
    gamma_R=.999       regularisation-dominated attenuation/reversal

The key mechanism is not a monotone change in hub share. Interior persistence
reduces attention turnover and strengthens common exposure, pairwise action
covariance, and aggregate-flow variance. Near gamma_R=.999, raw local reputation
dispersion falls well below the fixed sigma_0 scale, effective standardised
reputation differences shrink, and the propagation mechanism partially
reverses. Return-volatility differentiation follows this pattern but remains
small in level terms. Threshold-exceeding is zero for all topologies at every
gamma_R.

This interaction between persistence and the fixed reputation-standardisation
floor motivates D049.

## D049 joint alpha-beta-gamma_R interaction experiment — FROZEN AND IMPLEMENTED, TEST PENDING

Canonical protocol:

    docs/D049_JOINT_INTERACTION_PROTOCOL.md
    src/experiments/refined/joint_interaction_protocol.py

Frozen reduced factorial:

    alpha=(.40,.85,.99)
    beta=(0,1,5,100)
    gamma_R=(0,.90,.99,.999)

Continuity/control cells:

    C_ALPHA0 = (0,5,.90)
    A_D043   = (.75,1,.90)

Design:

    48 factorial cells + 2 controls = 50 cells
    experiment seed=2026091003
    bootstrap seed=2026091004
    paired replications per cell=300
    topology triplet=(R,SW,SF)
    checkpoints=15000
    simulations=45000
    bootstrap draws=5000 complete 50-cell x 3-topology blocks
    sigma_0 fixed=.0005

D049 is sequential exploratory interaction mapping because its factor levels
were selected after D046--D048 outcomes. No new Holm/FWER family is declared.

Primary signed interaction topology estimand:

    D(alpha,beta,gamma_R;Y) = Y_SF - Y_SW

The protocol freezes six interaction contrasts and three boundary-shift
contrasts. The priority mechanism-to-market chain is:

    mean_attention_overlap
      -> mean_pairwise_action_covariance
      -> mean_aggregate_order_flow_variance
      -> mean_absolute_order_flow_per_agent
      -> return_volatility
      -> peak_cid

Implementation:

    src/experiments/refined/joint_interaction_protocol.py
    src/experiments/refined/joint_interaction_analysis.py
    src/experiments/refined/joint_interaction_runner.py
    src/experiments/refined/joint_interaction_production.py
    scripts/run_refined_joint_interaction.py
    scripts/run_refined_joint_interaction.slurm
    scripts/finalize_refined_joint_interaction.py
    scripts/finalize_refined_joint_interaction.slurm

The finalizer refuses partial designs and verifies:

    cross-cell common random numbers
    exact alpha=0 economic-path topology null

before writing final artifacts.

## Immediate gate

1. Pull latest `refined-model`.
2. Run `python -m pytest -q tests/test_refined_*.py`.
3. Expected count is 800 if the full new layer passes; verify rather than assume.
4. Confirm working tree clean.
5. Do not submit D049 production yet.
6. First run a compute-node smoke for cell indexes:

       48  C_ALPHA0          (0,5,.90)
       49  A_D043            (.75,1,.90)
       26  interior          (.85,5,.99)
       47  boundary          (.99,100,.999)

7. Only after the smoke passes, create a fresh
   `results/refined/joint_interaction` directory and submit the 300-task array.
8. Do not inspect partial D049 outcome surfaces.

## Development status

    Phase 1   Refined fixed-topology core                     COMPLETE
    Phase 2   Topology generators                            COMPLETE
    Phase 3   Deterministic integration                      COMPLETE
    Phase 4   Paired design + structural validation          COMPLETE
    Phase 5   Market metrics / CID / calibration             COMPLETE
    Phase 6   Mechanism diagnostics                          COMPLETE
    Phase 7   Frozen baseline + market calibration           COMPLETE
    Phase 8   Paired confirmatory market runner              COMPLETE
    Phase 8b  D045 confirmatory production                   COMPLETE
    Phase 9a  D046 alpha sweep                               COMPLETE
    Phase 9b  D047 beta sweep                                COMPLETE
    Phase 9c  D048 gamma_R sweep                             COMPLETE
    Phase 9d  D049 joint alpha-beta-gamma_R interactions     IMPLEMENTED / TEST PENDING
    Phase 9e  Heterogeneity                                  PLANNED
    Phase 10  Endogenous G formation                         PLANNED
    Phase 11  Full Jacobian / Lyapunov                       PLANNED
    Phase 12  State-space / EKF / empirical work             PLANNED
    Phase 13  Planner / policy                               PLANNED
