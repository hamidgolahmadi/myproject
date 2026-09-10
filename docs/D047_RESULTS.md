# D047 Exploratory Beta Sweep — Final Results

Status: **COMPLETE**

D047 is an exploratory OAT regime-mapping exercise at the fixed D046-selected
social-weight anchor. It is not a second confirmatory multiple-testing family.

## Frozen design

- alpha anchor: 0.85
- beta grid: (0.00, 0.01, 0.10, 0.50, 1.00, 2.00, 5.00, 10.00, 100.00, 1000.00)
- paired replications per beta: 300
- topology triplet: (R, SW, SF)
- total treatment records / simulations: 9000
- bootstrap draws: 5000
- bootstrap seed: 2026090702
- confidence level: 95%
- cross-beta common random numbers: verified

The bootstrap unit is the complete replication block containing all ten beta
values and all three topologies.

## Final execution provenance

Successful production:

- Slurm array: 1536715
- 60/60 array tasks COMPLETED, exit code 0:0
- checkpoints: 3000/3000
- successful completion markers: 60
- non-empty stderr files: 0
- production commit: 6c94d6b92014185c4ed4799d7846b284b31f8a27

Finalization:

- Slurm job: 1537380
- COMPLETED, exit code 0:0
- stderr empty
- finalizer commit: 6c94d6b92014185c4ed4799d7846b284b31f8a27

Final artifacts:

- beta_sweep_records.csv
- beta_sweep_metadata.json
- beta_sweep_analysis.json
- beta_topology_means.csv
- beta_topology_gaps.csv
- beta_pairwise_contrasts.csv

## Numerical-validation incidents before final production

Two incomplete production attempts exposed validation-only clipping problems at
high beta. Neither was interpreted as an economic result.

The first problem occurred because valid tiny positive graph-supported attention
weights were zeroed before a row-stochasticity check. The second occurred because
valid tiny positive realised source-influence shares were zeroed after their sum
had already been verified. Both operations could remove accumulated probability
mass when softmax attention became highly concentrated.

The fixes preserve valid positive probability mass and do not change the economic
transition, frozen parameters, beta grid, shock process, or softmax equation.
After the second fix, 724 refined tests passed and a dedicated end-to-end smoke
job (1536571) completed successfully for beta={10,100,1000} before the final
production was restarted from a clean result directory.

## Main scientific result

Topology differences exist even at beta=0 because alpha=0.85 remains positive
and uniform graph-supported attention is still topology-specific. Reputation
selectivity is therefore not required for topology to matter.

The beta response is non-monotone at low selectivity and then shows a clear
amplification regime:

- beta <= about 0.5: weak selectivity; some market topology gaps narrow slightly;
- beta around 1--10: transition and strong amplification of topology-dependent
  common exposure and aggregate-flow fluctuations;
- beta around 10--100: continued amplification;
- beta around 100--1000: near-plateau / high-selectivity saturation for several
  market gap measures.

Selected relative topology gaps (max topology mean minus min topology mean,
relative to the cross-topology mean) illustrate the pattern:

| beta | return volatility | mean abs. order flow | aggregate OF variance |
| ---: | ---: | ---: | ---: |
| 0 | 0.590% | 1.134% | 4.238% |
| 1 | 0.505% | 0.772% | 3.880% |
| 2 | 0.692% | 1.051% | 5.651% |
| 5 | 1.008% | 1.920% | 8.997% |
| 10 | 1.039% | 1.931% | 9.851% |
| 100 | 1.305% | 2.795% | 11.463% |
| 1000 | 1.315% | 2.846% | 12.034% |

The main mechanism is better described as common exposure / synchronisation than
as a simple increase in structural-hub dominance. Structural hub influence
remains strongly topology-specific, but its relative topology gap is broadly
flat-to-declining with beta. In contrast, attention overlap rises strongly,
followed by pairwise action covariance and aggregate order-flow variance.

A concise mechanism chain is:

    stronger reputational selectivity
        -> more common/selective attention
        -> higher action covariance
        -> higher aggregate order-flow variance
        -> modestly larger realised market topology gaps

Threshold-exceeding is zero for all topologies at every D047 beta value. The
frozen calibration therefore shows amplification of the propagation mechanism,
not threshold-level market instability.

Peak CID rises in level with beta, but its cross-topology ordering is less robust
than the order-flow mechanism metrics. At high beta, SW is generally below R/SF,
while R and SF are not cleanly separated by their 95% matched-block intervals.

## Frozen thesis figure plan

Canonical plotting code:

    src/experiments/refined/beta_sweep_plotting.py
    scripts/plot_refined_beta_sweep.py

The final main-text figure set is deliberately small:

1. Peak CID topology levels with 95% matched-block intervals.
2. Core market relative-gap summary: return volatility, mean absolute order flow,
   and Peak CID.
3. Aggregate order-flow-variance relative topology gap, separately, with 95% CI.
4. Mechanism relative-gap summary: structural-hub influence and attention overlap.
5. Absolute topology gap in pairwise action covariance, with 95% CI.

Aggregate order-flow variance is separated from the other market gaps because its
relative gap is much larger and otherwise visually compresses the remaining
series.

Three x-axis representations remain available:

- `symlog`: preferred thesis view; actual beta values, including beta=0, with a
  small linear neighbourhood around zero and logarithmic spacing thereafter;
- `grid`: audit/full-design view with all predeclared beta values at equal visual
  spacing;
- `log`: positive-beta-only logarithmic view for transition-region inspection.

The plotting titles are thesis-style and do not use the internal `D047:` prefix.
Generated figure files retain `d047_` in their filenames for provenance.

The default command now generates only the preferred symlog main-text set in PNG
and vector PDF:

    python scripts/plot_refined_beta_sweep.py

For audit/appendix views:

    python scripts/plot_refined_beta_sweep.py --modes grid log symlog

For additional single-metric CI figures:

    python scripts/plot_refined_beta_sweep.py --include-detail

By default figures are written inside:

    results/refined/beta_sweep/figures/

which keeps generated images outside version control while keeping the plotting
code and scientific inputs reproducible.

After the final plotting test gate and one visual sanity check, D047 plotting is
closed. The next scientific task is Phase 9c: design and run the gamma_R sweep
before moving to heterogeneity and later joint parameter interactions.
