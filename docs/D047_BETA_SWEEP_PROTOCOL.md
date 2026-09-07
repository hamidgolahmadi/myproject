# D047 Exploratory Beta Sweep Protocol

Status: FROZEN BEFORE D047 OUTCOME INSPECTION

Date: 2026-09-07

## Purpose

D046 showed that alpha=0.85 lies in a coherent pre-boundary region where the
SF > R > SW mechanism is strong while the high-alpha ranking reversals at
0.95--0.99 have not yet appeared. D047 therefore holds alpha fixed at 0.85 and
maps how homogeneous reputational selectivity beta changes topology-dependent
influence concentration, correlated trading, aggregate order flow, and market
outcomes.

D047 is explicitly exploratory/OAT under D027. The alpha=0.85 anchor was chosen
after inspecting D046, so D047 must not be relabelled as an independent
confirmatory test.

## Scientific basis for the beta grid

The doctoral report defines beta as the sensitivity of softmax attention to
relative reputation. For feasible sources j and k,

    w_ij / w_ik = exp{ beta (z_ij - z_ik) }.

The report's retained beta sensitivity exercise uses an approximately
logarithmic range from 10^-2 to 10^3, reports comparatively little topology
movement below beta about 1, and identifies beta about 2--5 as the region where
topology separation becomes visually pronounced in the pilot output.

D047 therefore freezes the following grid before refined D047 outcomes are
inspected:

    beta = (0.00, 0.01, 0.10, 0.50, 1.00, 2.00, 5.00, 10.00, 100.00, 1000.00)

The grid is intentionally non-uniform:

- beta=0 is an exact no-selectivity control. Adaptive attention then remains
  uniform over each feasible graph neighbourhood, so reputation differences do
  not reallocate influence.
- 0.01, 0.10, and 0.50 resolve the weak-selectivity region.
- 1.00 retains the frozen D043/D045 beta anchor.
- 2.00 and 5.00 resolve the report's principal transition region.
- 10, 100, and 1000 map increasingly concentrated high-selectivity behaviour
  over the report-scale range.

The max-shifted softmax implementation is numerically stable over this grid;
large beta may legitimately produce near-degenerate attention allocations and
is treated as an economic boundary behaviour rather than a numerical target to
be clipped silently.

## Frozen design

    experiment seed = 2026090701
    alpha anchor = 0.85
    beta grid = (0.00, 0.01, 0.10, 0.50, 1.00, 2.00, 5.00, 10.00, 100.00, 1000.00)
    paired replications per beta = 300
    topology triplet = (R, SW, SF)
    total simulations = 10 * 300 * 3 = 9000
    bootstrap seed = 2026090702
    bootstrap draws = 5000
    confidence level = 0.95
    relative denominator epsilon = 1e-12

All parameters other than alpha and beta remain at the frozen D043 values.
Alpha is fixed at the D046-selected value 0.85 for every D047 treatment. Gamma_R
remains 0.9. D042/D044 reference scales and CID threshold remain fixed and are
not recalibrated over beta.

R=300 is retained from D046 as an exploratory computation choice. It gives a
worst-case Bernoulli Monte Carlo standard error of about 2.89 percentage points.
Continuous paired topology contrasts benefit materially from common random
numbers. Any later focused confirmatory experiment must use a new seed namespace
and a separately frozen design.

## Common-random-number design

For each replication id, the same semantic experiment namespace is reused over
the full beta grid. Beta does not enter the exogenous shock distribution or the
neutral non-network initial-state distribution. Therefore each replication
preserves across beta:

    shock innovations
    neutral initial state
    R/SW/SF graph seeds and graph realisations

The exact parameter fingerprint changes with beta, as it should, while the
semantic random-number plan remains matched.

One bootstrap unit is the complete replication block containing all ten beta
values and all three topology treatments. Independent resampling by beta or by
topology is prohibited.

## Outcomes

D047 reuses the D045 market, CID, mechanism, and exploratory outcome set. At
each beta it reports:

    topology means
    absolute and relative topology gaps where meaningful
    R-SW, R-SF, and SW-SF paired contrasts
    matched-block 95% percentile bootstrap intervals

No Holm/FWER rejection family is declared for D047. This is curve/regime
mapping, not a second confirmatory family.

Primary attention should be paid to:

    mean_hub_influence_share
    mean_influence_hhi
    mean_attention_entropy
    mean_effective_sources
    mean_attention_overlap
    mean_attention_mobility
    mean_pairwise_action_covariance
    mean_aggregate_order_flow_variance
    mean_absolute_order_flow_per_agent
    return_volatility
    peak_cid
    threshold_exceeding

The key empirical question is whether increasing beta primarily strengthens the
early upstream influence-concentration channel, and how much of that additional
concentration survives downstream into aggregate order flow and price outcomes.

## Interpretation guard

The report's pilot suggests that larger beta can strengthen topology-dependent
differences, especially in hub-dominated networks, but D047 does not assume
monotonicity. The refined run may show activation, saturation, ranking reversal,
or weak downstream translation.

A later gamma_R sweep or joint beta-by-gamma_R experiment may use the complete
D047 curves to choose focused regions, but must use a new independent seed
namespace and must document any post-D047 selection explicitly.

## Persistence and partial-result guard

Each checkpoint is one indivisible R/SW/SF triplet for one
`(beta, replication_id)` pair and is bound to the complete D047/D043/D044
configuration fingerprint.

Final artifacts are written only when all 10 x 300 = 3000 beta/replication
checkpoints validate.

Canonical planned artifacts:

    results/refined/beta_sweep/beta_sweep_records.csv
    results/refined/beta_sweep/beta_sweep_metadata.json
    results/refined/beta_sweep/beta_sweep_analysis.json
    results/refined/beta_sweep/beta_topology_means.csv
    results/refined/beta_sweep/beta_topology_gaps.csv
    results/refined/beta_sweep/beta_pairwise_contrasts.csv

No partial beta/topology curve should be inspected or used to change the frozen
D047 grid, replication count, D043 market baseline, D044 calibration, or D047
analysis definitions.
