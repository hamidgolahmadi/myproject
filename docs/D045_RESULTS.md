# D045 First Confirmatory Results

Status: VERIFIED AND FROZEN

Date: 2026-09-04

This document records the first final fixed-topology confirmatory outcome set.
It is descriptive of the completed D045 experiment and must not be used to
retune D043/D044 calibration inputs.

## Provenance

- production Slurm array: `1511972`
- finalization Slurm job: `1512116`
- compute commit for all 10 production tasks and finalizer:
  `b5fbf52dd988637d90d7b5bc5c346c20551b66be`
- all 10 array tasks: `COMPLETED`, exit code `0:0`
- stderr: empty for every array task
- paired replications: `1000`
- treatment records: `3000`
- production seed: `2026090402`
- bootstrap draws: `10000`
- bootstrap seed: `2026090403`
- refined test gate after the final test-only compatibility fix: `660 passed`

Final artifacts:

    results/refined/confirmatory_production/confirmatory_records.csv
    results/refined/confirmatory_production/confirmatory_metadata.json
    results/refined/confirmatory_production/confirmatory_analysis.json
    results/refined/confirmatory_production/topology_means.csv
    results/refined/confirmatory_production/topology_gaps.csv
    results/refined/confirmatory_production/pairwise_contrasts.csv

## Primary results

Topology means:

| Outcome | R | SW | SF |
|---|---:|---:|---:|
| return volatility | 0.00353188 | 0.00352292 | 0.00353480 |
| RMS mispricing | 0.06949164 | 0.06950047 | 0.06950268 |
| maximum absolute mispricing | 0.22252408 | 0.22246876 | 0.22256315 |
| mean absolute order flow / agent | 0.07757663 | 0.07737409 | 0.07773647 |
| peak CID | 1.20911592 | 1.20776649 | 1.20968369 |
| threshold-exceeding rate | 0.001 | 0.002 | 0.002 |

After Holm control over the 18 primary pairwise hypotheses:

- return volatility: all three topology contrasts reject;
- mean absolute order flow per agent: all three topology contrasts reject;
- peak CID: `R-SW` and `SW-SF` reject, `R-SF` does not;
- RMS mispricing: no topology contrast rejects;
- maximum absolute mispricing: no topology contrast rejects;
- threshold-exceeding rate: no topology contrast rejects.

The realised ordering in return volatility and mean absolute order flow is:

    SF > R > SW

The effect is statistically precise but economically modest at the frozen
baseline. For example, the SF-SW return-volatility difference is about 0.34%
of the SW mean, while the SF-SW mean-absolute-order-flow difference is about
0.47% of the SW mean.

## Mechanism results

Topology means:

| Outcome | R | SW | SF |
|---|---:|---:|---:|
| mean hub influence share | 0.09408866 | 0.06027819 | 0.18796179 |
| mean attention overlap | 0.01383593 | 0.01143207 | 0.02529628 |
| mean pairwise action covariance | 0.00728987 | 0.00717841 | 0.00734232 |
| mean aggregate order-flow variance | 73.38284775 | 72.28983878 | 73.90365082 |

All 12 predeclared mechanism pairwise contrasts reject after the separate Holm
family-wise correction. The realised ordering is consistently:

    SF > R > SW

This provides strong evidence for the maintained mechanism chain:

    topology
      -> realised influence concentration
      -> attention overlap
      -> correlated actions
      -> aggregate order-flow amplification
      -> price response

However, the amplification attenuates strongly as it moves toward final price
outcomes. D045 therefore supports the statement that topology reorganises and
amplifies the social-transmission mechanism, while topology alone at
`alpha=0.75` is not sufficient to generate large operational instability.

## Interpretation guard

Do not state that the hub-dominated topology mechanically generates severe
market instability. Under D045, threshold exceedance is rare and not
significantly different across topology classes, and mispricing differences do
not survive family-wise correction.

The next research task is an explicitly exploratory OAT alpha sweep to map
where topology differentiation is weak, strongest, and potentially saturated.
