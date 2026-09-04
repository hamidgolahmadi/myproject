# D046 Exploratory Alpha Sweep Protocol

Status: FROZEN BEFORE D046 OUTCOME INSPECTION

Date: 2026-09-04

## Purpose

D045 established a strong topology-to-mechanism effect at the frozen baseline
`alpha=0.75`, but only modest realised price-instability differences. D046 asks
where along the social-weight parameter alpha this topology differentiation is
weak, strongest, or saturated.

D046 is an explicitly exploratory one-factor-at-a-time diagnostic under D027.
It does not replace the later joint-parameter identification design.

## Frozen design

    experiment seed = 2026090404
    alpha grid = (0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 1.00)
    paired replications per alpha = 300
    topology triplet = (R, SW, SF)
    total simulations = 8 * 300 * 3 = 7200
    bootstrap seed = 2026090405
    bootstrap draws = 5000
    confidence level = 0.95

All parameters other than alpha remain at the frozen D043 values. D042/D044
reference scales and CID threshold remain fixed and are not recalibrated across
alpha.

The grid deliberately includes:

- `alpha=0.00`: exact no-social negative-control endpoint;
- `0.20, 0.40`: weak-to-moderate social transmission;
- `0.60`: intermediate transmission below the D043 anchor;
- `0.75`: the D043/D045 baseline anchor;
- `0.85, 0.95`: high social transmission where synchronisation may strengthen;
- `1.00`: boundary endpoint where the contemporaneous private-signal weight
  `(1-alpha)` is zero.

The grid is non-uniform because the report's conceptual regime discussion
places particular interest on possible high-alpha synchronisation/saturation.
No alpha value is selected because of D046 outcomes; this grid is frozen before
running D046.

## Common-random-number design

For each replication id, the same semantic experiment namespace is used at all
alpha values. Because alpha does not enter the exogenous shock distributions or
neutral initial-state distribution, this produces the same:

    shock innovations
    neutral initial state
    R/SW/SF graph seeds

across the alpha grid within a replication. The parameter fingerprint still
changes with alpha and remains validated by the paired-plan guard.

Thus one replication is a matched 8-alpha x 3-topology block. Bootstrap
resampling must preserve that complete block. Independent resampling by alpha
or topology is prohibited.

## Outcomes

D046 reuses the D045 treatment record and therefore carries the same primary,
mechanism, and secondary metrics. D046 is exploratory: it reports topology
means, topology gaps, and all three pairwise topology contrasts at each alpha,
with matched-block percentile bootstrap intervals.

No Holm/FWER rejection family is declared for D046. The purpose is curve and
regime mapping, not a second confirmatory multiple-testing exercise.

Particular attention should be paid to the following continuous curves:

    return_volatility
    mean_absolute_order_flow_per_agent
    peak_cid
    mean_hub_influence_share
    mean_attention_overlap
    mean_pairwise_action_covariance
    mean_aggregate_order_flow_variance

`threshold_exceeding` remains reported, but its low baseline frequency means it
must not be the sole selector of an alpha regime in this exploratory stage.

## Interpretation

D046 may reveal monotone amplification, an interior peak, saturation, or even
ranking reversals. None is assumed in advance.

A candidate region for later focused testing should be chosen from the complete
D046 curves, documented explicitly, and then evaluated with a new independent
seed namespace in a later decision. D046 itself must not be relabelled as a
confirmatory test after its outcomes are seen.

## Persistence and partial-result guard

Each checkpoint is one indivisible R/SW/SF triplet for one `(alpha,
replication_id)` pair and is bound to the complete D046/D043/D044
configuration fingerprint.

Final D046 artifacts are written only when every alpha has all 300 complete
paired checkpoints.

Canonical planned artifacts:

    results/refined/alpha_sweep/alpha_sweep_records.csv
    results/refined/alpha_sweep/alpha_sweep_metadata.json
    results/refined/alpha_sweep/alpha_sweep_analysis.json
    results/refined/alpha_sweep/alpha_topology_means.csv
    results/refined/alpha_sweep/alpha_topology_gaps.csv
    results/refined/alpha_sweep/alpha_pairwise_contrasts.csv

No partial curve should be used to alter the frozen grid, replication count,
D043 baseline, D044 calibration, or D046 analysis definitions.
