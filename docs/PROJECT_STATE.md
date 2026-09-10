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

These remain immutable through D046--D048.

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

Frozen design:

    alpha=(0,.2,.4,.6,.75,.85,.95,.99)
    300 paired replications per alpha
    7200 simulations
    bootstrap=5000 complete blocks

Execution:

    array=1526697
    checkpoints=2400/2400
    finalizer=1527139
    commit=5283a338d75560f27d28a19d48e07f86cdeada07

Main result: absolute activity/volatility generally falls with alpha while
cross-topology differentiation strengthens into the coherent pre-boundary
region; alpha=.95--.99 displays a qualitatively different boundary regime.
Canonical results: `docs/D046_RESULTS.md`.

## D047 exploratory beta sweep — COMPLETE

Frozen design:

    alpha anchor=.85
    beta=(0,.01,.1,.5,1,2,5,10,100,1000)
    300 paired replications per beta
    9000 simulations
    bootstrap=5000 complete 10-beta x 3-topology blocks

Execution:

    production array=1536715
    60/60 tasks COMPLETED 0:0
    checkpoints=3000/3000
    non-empty stderr=0
    finalizer=1537380 COMPLETED 0:0
    production/finalizer commit=6c94d6b92014185c4ed4799d7846b284b31f8a27

Scientific result:

    beta <= about .5      weak-selectivity region
    beta about 1--10      clear amplification transition
    beta about 10--100    continued amplification
    beta about 100--1000  near-plateau for several market gaps

Topology already matters at beta=0 because alpha=.85 and uniform attention is
still graph-specific. The strongest D047 mechanism is common exposure / overlap
-> action covariance -> aggregate order-flow variance, not a monotone increase
in structural-hub dominance. Threshold-exceeding is zero throughout D047.
Canonical results: `docs/D047_RESULTS.md`.

D047 thesis plotting is closed. Preferred main-text x-axis is symlog, retaining
beta=0 and resolving the logarithmic high-beta range. The default plotting
command generates five main figures in PNG and vector PDF (10 files total).

## D048 exploratory gamma_R sweep — COMPLETE AND FINALIZED

Canonical protocol:

    docs/D048_GAMMA_SWEEP_PROTOCOL.md
    src/experiments/refined/gamma_sweep_protocol.py

Frozen design:

    experiment seed=2026091001
    bootstrap seed=2026091002
    alpha anchor=.85
    beta anchor=5.0
    gamma_R=(0,.5,.8,.9,.95,.98,.99,.995,.999)
    paired replications per gamma=300
    matched blocks=2700
    simulations=8100
    bootstrap draws=5000
    confidence=.95

D048 preserves the full D045/D047 outcome set and adds:

    mean_raw_local_reputation_std
    mean_raw_local_reputation_std_over_sigma0

These distinguish a direct persistence effect from an indirect change in the
effective strength of the fixed sigma_0 regularisation floor.

Validation and execution provenance:

    full refined test gate: 766 passed in 32.42s
    smoke job: 1541812
    smoke gamma_R={0,.9,.999}
    smoke tasks: 3/3 COMPLETED 0:0
    smoke checkpoints: 3
    smoke non-empty stderr: 0

    production array: 1542666
    production tasks: 54/54 COMPLETED 0:0
    production checkpoints: 2700/2700
    production completion markers: 54
    production non-empty stderr: 0
    production commit across all 54 tasks:
        ce3630faf86d7f55b3535d2a8b902d5c677a6f0c

    finalizer: 1544241
    finalizer state: COMPLETED 0:0
    finalizer host: ruby047
    finalizer elapsed: 00:00:10
    finalizer stderr: empty
    finalizer commit:
        ce3630faf86d7f55b3535d2a8b902d5c677a6f0c

Final artifacts:

    results/refined/gamma_sweep/gamma_sweep_records.csv
    results/refined/gamma_sweep/gamma_sweep_metadata.json
    results/refined/gamma_sweep/gamma_sweep_analysis.json
    results/refined/gamma_sweep/gamma_topology_means.csv
    results/refined/gamma_sweep/gamma_topology_gaps.csv
    results/refined/gamma_sweep/gamma_pairwise_contrasts.csv

No D048 scientific result has yet been documented in this file. Interpretation
must begin from the finalized artifacts only.

## Immediate scientific task

Extract and inspect finalized D048 outcomes in this order:

    reputation dispersion / sigma_0 ratio
    attention mobility / entropy / effective sources / HHI / overlap
    action covariance / aggregate order-flow variance
    mean absolute order flow / return volatility
    Peak CID / threshold exceedance / mispricing

D048 is exploratory OAT. Use matched-block 95% intervals and do not relabel
pointwise evidence as a confirmatory family-wise test.

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
    Phase 9c  D048 gamma_R sweep                             COMPLETE / RESULTS PENDING INTERPRETATION
    Phase 9d  Heterogeneity                                  PLANNED
    Phase 10  Endogenous G formation                         PLANNED
    Phase 11  Full Jacobian / Lyapunov                       PLANNED
    Phase 12  State-space / EKF / empirical work             PLANNED
    Phase 13  Planner / policy                               PLANNED
