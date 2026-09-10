# D048 Exploratory Gamma_R Sweep Protocol

Status: **FROZEN BEFORE D048 OUTCOME INSPECTION**

Date: 2026-09-10

## Purpose

D048 maps how persistence in performance-based reputation changes attention
persistence, common exposure, correlated trading, aggregate order flow, and
market outcomes. It follows completed D046 alpha and D047 beta regime mapping.

The reputation transition is

    R_i,t = gamma_R R_i,t-1 + (1-gamma_R) pi_i,t.

Thus gamma_R controls memory/persistence, whereas beta controls the sensitivity
of softmax attention to cross-sectional reputation differences.

D048 is explicitly exploratory/OAT under D027. The alpha and beta anchors were
selected after D046/D047 and must not be presented as independently pre-specified
confirmatory choices.

## Frozen anchors

D048 holds

    alpha = 0.85
    beta  = 5.0

Alpha=0.85 is the coherent pre-boundary social-weight anchor selected after
D046. Beta=5 is selected after D047 as an interior amplification point: it lies
inside the region where reputational selectivity materially strengthens common
exposure and aggregate-flow differentiation but is not in the extreme
beta=100--1000 near-saturation range.

All other D043 parameters remain fixed, including sigma_0=0.0005. D042/D044
reference scales and the CID threshold remain fixed and are not recalibrated.

## Frozen gamma_R grid

    gamma_R = (0.000, 0.500, 0.800, 0.900, 0.950, 0.980, 0.990, 0.995, 0.999)

The grid is deliberately dense near one because the economically relevant
innovation weight is 1-gamma_R and the reputation-memory horizon becomes highly
nonlinear near the upper boundary.

Interpretation of selected points:

- gamma_R=0 is an exact **no-memory** control: R_i,t = pi_i,t. It is not a
  no-reputation control.
- gamma_R=0.9 retains the frozen D043 baseline reputation-persistence anchor.
- gamma_R=0.95--0.995 resolve increasingly persistent reputation rankings.
- gamma_R=0.999 is a horizon-scale persistence stress point. Its geometric
  half-life is about 693 periods when T=1000. It is not a no-learning control.
- gamma_R=1 is excluded because the maintained model requires 0 <= gamma_R < 1.

Approximate geometric half-lives, log(0.5)/log(gamma_R), are about 1 period at
0.5, 3.1 at 0.8, 6.6 at 0.9, 13.5 at 0.95, 34.3 at 0.98, 69 at 0.99, 138 at
0.995, and 693 at 0.999.

## Frozen production design

    experiment seed = 2026091001
    bootstrap seed = 2026091002
    paired replications per gamma_R = 300
    topology triplet = (R, SW, SF)
    total matched gamma/replication blocks = 9 * 300 = 2700
    total simulations = 9 * 300 * 3 = 8100
    bootstrap draws = 5000
    confidence level = 0.95
    relative denominator epsilon = 1e-12

The D048 seed namespaces are disjoint from D042, D045, D046, and D047.

## Common-random-number design

Within each replication id, the same semantic experiment namespace is reused
over the full gamma_R grid. Gamma_R does not alter the exogenous shock
distribution or the neutral non-network initial-state distribution. Therefore
each replication preserves across gamma_R:

    shock innovations
    neutral initial state
    R/SW/SF graph seeds and graph realisations

The parameter fingerprint changes with gamma_R, as it should. One bootstrap unit
is the complete 9-gamma x 3-topology replication block. Independent resampling
by gamma_R or by topology is prohibited.

## Outcomes

D048 retains the full D045/D047 market, CID, mechanism, and exploratory outcome
set. It additionally records two gamma-specific diagnostics:

    mean_raw_local_reputation_std
    mean_raw_local_reputation_std_over_sigma0

For each agent i and period t, raw local reputation dispersion is

    sigma_raw_i,t = sqrt(mean_{j in N_i} (R_j,t - mean_{k in N_i} R_k,t)^2).

It deliberately excludes sigma_0. The run-level diagnostic averages this raw
quantity over t=1,...,T and agents. Dividing by the fixed sigma_0 exposes whether
changes in gamma_R alter the effective strength of the regularisation floor.

These diagnostics are essential because a large gamma_R can reduce the scale of
new reputation innovations. If raw reputation dispersion becomes small relative
to sigma_0, effective standardised reputation differences can shrink even when
memory itself is high. D048 must therefore distinguish a direct persistence
mechanism from attenuation through the fixed regularisation floor.

At each gamma_R, report:

    topology means
    absolute and relative topology gaps where meaningful
    R-SW, R-SF, and SW-SF paired contrasts
    matched-block 95% percentile bootstrap intervals

No Holm/FWER family is declared. D048 is exploratory regime mapping.

## Ex-ante interpretation guard

D048 does not assume monotonicity. Before outcomes are inspected, the admissible
mechanisms are:

    low gamma_R
        -> fast but noisy reputation updating
        -> potentially high attention turnover

    intermediate gamma_R
        -> noise filtering plus persistent ranking
        -> potentially stronger common exposure / synchronisation

    very high gamma_R
        -> slow learning / reputation inertia
        -> possible lock-in, plateau, or attenuation through sigma_0

Hump shapes, plateaus, reversals, or weak downstream translation are all
admissible. Do not retrofit a monotone story after inspecting the curves.

## Persistence and partial-result guard

Each checkpoint is one indivisible R/SW/SF triplet for one
`(gamma_index, replication_id)` pair and is bound to the complete
D048/D043/D044 configuration fingerprint.

Final artifacts are written only when all 9 x 300 = 2700 checkpoints validate.
No partial gamma/topology curve should be inspected or used to change the frozen
grid, replication count, anchors, baseline, CID calibration, or analysis
definitions.

Canonical planned artifacts:

    results/refined/gamma_sweep/gamma_sweep_records.csv
    results/refined/gamma_sweep/gamma_sweep_metadata.json
    results/refined/gamma_sweep/gamma_sweep_analysis.json
    results/refined/gamma_sweep/gamma_topology_means.csv
    results/refined/gamma_sweep/gamma_topology_gaps.csv
    results/refined/gamma_sweep/gamma_pairwise_contrasts.csv

## Execution sequence

1. Pass the full refined test gate.
2. Run a tiny end-to-end compute-node smoke at gamma_R={0,0.9,0.999}.
3. If the smoke is clean, create a fresh `results/refined/gamma_sweep` directory.
4. Submit the 54-task production Slurm array.
5. Do not inspect partial outcome curves.
6. Verify 2700 checkpoints, 54 successful tasks, empty stderr, and one commit.
7. Run the separate finalizer only after the complete design validates.
