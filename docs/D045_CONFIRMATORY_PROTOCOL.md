# D045 — First Confirmatory Fixed-Topology Production Protocol

Status: **FROZEN BEFORE PRODUCTION OUTCOME INSPECTION**

This document records the first production design for the paired Random (R),
Small-World (SW), and hub-dominated / Scale-Free-style (SF) benchmark comparison.
It is frozen after the successful Phase-8 end-to-end smoke and before any D045
production topology outcomes are inspected.

## Fixed scientific inputs

D045 uses without retuning:

- D041 benchmark graph design: `N=100`, `K=6`, `q=5`, `p_sw=0.02`, `a0=1.0`;
- D043 homogeneous market specification with baseline `alpha=0.75`;
- D044 frozen market-evaluation scales and CID threshold;
- the D025 common-random-number paired treatment design.

The production experiment is baseline-only. The exact `alpha=0` topology-null
property has already been established algebraically, by deterministic regression
tests, and by the successful Phase-8 Iridis smoke (job 1509863), in which both
paired replications produced one unique non-attention economic-path fingerprint
across R/SW/SF. Repeating 1000 alpha-zero triplets would add computational cost
without adding information about a treatment effect that is exactly zero by
construction. Alpha zero remains part of later alpha-sweep work.

## Production Monte Carlo design

    production experiment seed = 2026090402
    paired replications         = 1000
    topology treatments         = (R, SW, SF)
    simulations                 = 3000
    bootstrap seed              = 2026090403
    bootstrap draws             = 10000
    confidence level            = 95%
    family-wise alpha           = 0.05

The production and bootstrap namespaces are disjoint from the Phase-8 smoke
namespace `2026090401` and from every D042 calibration namespace.

The doctoral report defines the paired estimator and requires complete matched
R/SW/SF replication triplets to be resampled together in the bootstrap. It does
not prescribe a numerical production replication count or bootstrap count.
`R=1000` and `B_boot=10000` are therefore explicit pre-production design choices,
not report equations.

For a binary run-level probability, `R=1000` gives a worst-case Monte Carlo
standard error of approximately

    sqrt(0.25 / 1000) = 0.0158,

or 1.58 percentage points. This provides substantially tighter Monte Carlo
precision than a small pilot while remaining practical on Iridis.

## Pairwise topology contrasts

All three named contrasts are predeclared and retain the subtraction order:

    R - SW
    R - SF
    SW - SF

For every continuous/non-negative outcome, report topology means, the absolute
topology gap, the relative topology gap, each named pairwise contrast, and the
relative pairwise effect. The fixed denominator regulariser is `1e-12` and is
identical across topology classes.

Relative effects are not used for binary outcomes or for signed average
pairwise action covariance.

## Primary confirmatory family

The first family contains six market/CID outcomes:

1. return volatility;
2. RMS mispricing;
3. maximum absolute mispricing;
4. mean absolute signed net order flow per agent;
5. peak CID;
6. threshold-exceedance indicator / rate.

With three topology pairs this produces 18 predeclared pairwise hypotheses.
Holm family-wise error control at 5% is applied across these 18 hypotheses.
Bootstrap percentile intervals are also reported for each estimate.

## Mechanism confirmatory family

The second family contains four intermediate mechanism outcomes:

1. mean realised influence received by the structural top-q hubs;
2. mean attention overlap;
3. mean pairwise action covariance;
4. mean aggregate signed-order-flow variance.

With three topology pairs this produces 12 predeclared pairwise hypotheses.
Holm family-wise error control at 5% is applied separately across these 12
mechanism hypotheses.

This separation reflects the report's distinction between observing an outcome
difference and establishing the proposed transmission mechanism.

## Secondary / exploratory outcomes

The following are retained for interpretation and robustness but are not part of
the two confirmatory multiplicity families:

- mean absolute return;
- time-averaged cross-sectional belief variance;
- CID exceedance-duration share;
- right-censoring / non-stabilisation indicator;
- realised-influence HHI;
- mean attention entropy;
- mean effective number of sources;
- mean attention mobility;
- mean sum of individual action variances.

These receive pointwise bootstrap intervals and are explicitly labelled
exploratory. The right-censoring rate is the baseline censoring-aware
stabilisation summary; right-censored runs are not dropped or treated as
numerical failures.

## Bootstrap and uncertainty

Bootstrap resampling operates on complete replication identifiers. If replication
`r` is selected, its entire `(R, SW, SF)` triplet is carried into the bootstrap
draw. Topologies are never bootstrapped independently in this paired design.

For each draw the analysis recomputes topology means, absolute/relative gaps,
and all named pairwise contrasts. Pointwise 95% intervals use percentile
quantiles. Two-sided bootstrap p-values for pairwise contrasts are calculated
from the centered bootstrap distribution and then passed to the predeclared
Holm corrections for the primary and mechanism families.

## Persistence and partial-result guard

Each production replication is checkpointed as one indivisible triplet:

    results/refined/confirmatory_production/replications/replication_XXXX.json

Every checkpoint is bound to a SHA-256 fingerprint of the complete D045
protocol, frozen baseline, frozen market calibration, record schema, and frozen
D044 fingerprints. A mismatched or stale checkpoint fails loudly.

Final topology means, contrasts, gaps, and bootstrap inference are not written
until all 1000 predeclared paired checkpoints are present. The final artifacts
are:

    confirmatory_records.csv
    confirmatory_metadata.json
    confirmatory_analysis.json
    topology_means.csv
    topology_gaps.csv
    pairwise_contrasts.csv

This guard prevents an incomplete production sample from being presented or
implicitly interpreted as the final confirmatory result.

## Computational execution

The production sample is divided into a 10-task Slurm array with 100 paired
replications per task. Each task uses one CPU and checkpoints after every paired
triplet. Resubmission safely reuses only configuration-matched completed
checkpoints.

The inference/finalization step is a separate compute job and may run only after
all array tasks complete successfully.

## Additional implementation guard

`PairedReplicationPlan` records the agent dimension, horizon, and a SHA-256
fingerprint of the exact refined parameter vector used to generate its common
shock path. Treatment preparation rejects any attempt to reuse that plan with a
different parameter vector. This prevents silent violation of the paired design
when shock-generating parameters change.
