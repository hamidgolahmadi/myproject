# Final D042 Market-Evaluation Calibration

Status: FROZEN

This document records the final numerical calibration of the market-evaluation
CID used for the first confirmatory fixed-topology refined-model experiments.

The calibration method was frozen in D042 and the market specification in D043
before this production run was inspected.

## Production design

- calibration benchmark: no-social, `alpha = 0`;
- scale sample: 500 independent runs;
- scale seed namespace: `2026090201`;
- threshold sample: 500 independent runs;
- threshold seed namespace: `2026090202`;
- horizon: `T = 1000`;
- burn-in: `B = 0`;
- rolling window: `L = 50`;
- rolling endpoints per run: 951;
- reference scales: pooled medians of the raw rolling return, belief, and signed-order-flow components;
- CID weights: equal thirds;
- threshold: 95th percentile of run-level peak CID values;
- empirical quantile method: `higher`;
- component guardrails: inactive;
- stabilisation length: `L_stab = 50`.

At `alpha=0`, adaptive and fixed attention are exactly equivalent for the
market paths and CID components under common graph/shock/initial-state
randomness. The production calibration therefore used
`adaptive_attention=False` as a computational shortcut, after this exact
equivalence had been regression-tested.

## Frozen numerical values

    c_ret = 0.0030364359162156455
    c_bel = 0.004182211355781272
    c_F   = 0.11381404220614316
    c_CID = 1.8326578831721285

These values are now fixed inputs to the first confirmatory topology
experiments. They must not be retuned after inspecting Random, Small-World, or
Hub-dominated treatment outcomes.

## Reproducibility fingerprints

Configuration fingerprint:

    9200fcdd3fbfb60fe04d29e2978394b6575bd9538e3c23f62d8d04de5d862202

Reference-scales fingerprint:

    1e89574139dfe70e70742e98b1603b6d976fb85addce1eb9bbb21c04082ba476

The configuration fingerprint binds the calibration to the complete frozen
D042 protocol and D043 market specification. The reference-scales fingerprint
binds the three numerical normalisation scales.

## Iridis production provenance

Slurm job:

    1505911

Execution host:

    ruby047

Run timing:

    submit: 2026-09-03 10:03:48 BST
    start:  2026-09-03 10:04:28 BST
    end:    2026-09-03 11:11:04 BST
    elapsed: 01:06:36

Completion:

    State: COMPLETED
    exit code: 0
    cores: 1
    CPU utilised: 01:06:09
    CPU efficiency: 99.32%
    peak reported memory: 218.15 MB

Production artifact on Iridis:

    results/refined/market_calibration/market_evaluation_calibration.json

Threshold peak-CID audit file:

    results/refined/market_calibration/threshold_peak_cids.csv

The completed run produced exactly 500 scale checkpoints and 500 threshold
checkpoints.

## Interpretation and use

`c_ret`, `c_bel`, and `c_F` are diagnostic normalisation scales, not structural
market parameters. `c_CID` is an operational threshold calibrated from the
no-social run-level peak CID distribution. Crossing it does not by itself imply
mathematical divergence, non-stationarity, or failure of the equilibrium
Jacobian stability condition.

Any future recalibration must be treated as a new explicit methodological
decision, use disjoint calibration seeds, and be reported separately from this
first frozen calibration.
