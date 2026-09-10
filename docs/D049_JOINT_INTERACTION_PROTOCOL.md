# D049 Joint Interaction Experiment: alpha x beta x gamma_R

Status: **FROZEN BEFORE D049 OUTCOME INSPECTION**

Date: 2026-09-10

## Purpose

D049 moves beyond the one-factor-at-a-time maps in D046--D048 while retaining
the homogeneous fixed-G benchmark. The experiment asks whether the effect of
network architecture on realised influence and market propagation depends
jointly on social weight (`alpha`), reputational selectivity (`beta`), and
reputation persistence (`gamma_R`).

D049 is a sequential exploratory interaction experiment, not a new confirmatory
family. Its factor levels were selected after inspecting D046--D048. No
Holm/FWER claims are attached to D049; uncertainty is reported with matched
complete-block 95% percentile bootstrap intervals.

The main signed topology contrast is

    D(alpha,beta,gamma_R; Y) = Y_SF - Y_SW.

Positive D means the scale-free/hub-dominated treatment exceeds the
small-world treatment for outcome Y. Cellwise R-SW, R-SF, and SW-SF contrasts
are also retained for continuity with D045--D048.

## Frozen reduced factorial

The 48-cell factorial is

    alpha   = (0.40, 0.85, 0.99)
    beta    = (0.0, 1.0, 5.0, 100.0)
    gamma_R = (0.0, 0.90, 0.99, 0.999)

Interpretation of the selected levels:

- `alpha=0.40`: moderate social transmission.
- `alpha=0.85`: coherent high-social-weight regime identified in D046.
- `alpha=0.99`: high-social-weight boundary regime from D046.
- `beta=0`: exact no-selectivity attention control conditional on G.
- `beta=1`: D043 baseline selectivity.
- `beta=5`: interior D047 amplification point.
- `beta=100`: high-selectivity near-plateau/stress regime from D047.
- `gamma_R=0`: exact no-memory reputation control.
- `gamma_R=0.90`: D043 baseline persistence.
- `gamma_R=0.99`: high-persistence interior regime from D048.
- `gamma_R=0.999`: horizon-scale persistence boundary where D048 showed
  regularisation-floor attenuation.

Two non-factorial continuity/control cells are added:

    alpha0_control = (alpha=0.0, beta=5.0, gamma_R=0.90)
    d043_anchor     = (alpha=0.75, beta=1.0, gamma_R=0.90)

The alpha-zero cell is an exact network-propagation negative control for the
economic path. The D043 anchor provides continuity with the original frozen
homogeneous baseline. These two cells are not used inside factorial interaction
contrasts.

Total parameter cells:

    48 factorial + 2 control/anchor = 50.

## Frozen production design

    experiment seed = 2026091003
    bootstrap seed = 2026091004
    paired replications per cell = 300
    topology triplet = (R, SW, SF)
    parameter cells = 50
    matched cell/replication checkpoints = 50 * 300 = 15000
    total simulations = 50 * 300 * 3 = 45000
    bootstrap draws = 5000
    confidence level = 0.95
    relative denominator epsilon = 1e-12

D049 keeps every D043 parameter not explicitly varied above fixed. In
particular, `sigma_0=0.0005` remains fixed. The D042/D044 CID scales and
threshold remain immutable and are not recalibrated.

## Common-random-number design

Within replication r, every one of the 50 parameter cells shares the same:

    shock innovations
    neutral non-network initial state seed
    topology-specific R/SW/SF graph seeds and graph realisations

Only the parameter fingerprint changes across cells. This is valid because
alpha, beta, and gamma_R do not alter the exogenous shock distribution, graph
distribution, or neutral initial-state distribution.

One bootstrap unit is the complete replication block containing

    50 parameter cells x 3 topologies.

Independent resampling by cell or topology is prohibited.

## Outcome hierarchy

The full D045 outcome set is retained, together with the two D048 reputation
scale diagnostics:

    mean_raw_local_reputation_std
    mean_raw_local_reputation_std_over_sigma0

The core D049 mechanism-to-market chain is

    mean_attention_overlap
      -> mean_pairwise_action_covariance
      -> mean_aggregate_order_flow_variance
      -> mean_absolute_order_flow_per_agent
      -> return_volatility
      -> peak_cid

The remaining influence, entropy/effective-source, reputation-scale,
mispricing, individual-action-variance, and threshold/censoring outcomes are
retained as diagnostics/secondary outcomes.

## Frozen interaction estimands

Interactions are computed on the signed SF-SW contrast D, not on the unsigned
max-minus-min topology gap.

For a factor x, define a finite-difference operator

    Delta_x[a -> b] D = D(x=b) - D(x=a),

holding the other coordinates fixed as specified below.

The predeclared estimands are:

### Interior/core interactions

1. `beta_gamma_core_at_alpha085`

       Delta_beta[0 -> 5] Delta_gamma[0 -> .99] D
       at alpha=.85

2. `alpha_beta_core_at_gamma09`

       Delta_alpha[.40 -> .85] Delta_beta[0 -> 5] D
       at gamma_R=.90

3. `alpha_gamma_core_at_beta5`

       Delta_alpha[.40 -> .85] Delta_gamma[0 -> .99] D
       at beta=5

4. `alpha_beta_gamma_core`

       Delta_alpha[.40 -> .85]
       Delta_beta[0 -> 5]
       Delta_gamma[0 -> .99] D

### Boundary/regime interactions and shifts

5. `beta_gamma_boundary_at_alpha085`

       Delta_beta[5 -> 100] Delta_gamma[.99 -> .999] D
       at alpha=.85

6. `alpha_gamma_boundary_at_beta5`

       Delta_alpha[.85 -> .99] Delta_gamma[.99 -> .999] D
       at beta=5

7. `gamma_boundary_shift_at_alpha085_beta5`

       Delta_gamma[.99 -> .999] D
       at alpha=.85, beta=5

8. `alpha_boundary_shift_at_beta5_gamma09`

       Delta_alpha[.85 -> .99] D
       at beta=5, gamma_R=.90

9. `beta_saturation_shift_at_alpha085_gamma09`

       Delta_beta[5 -> 100] D
       at alpha=.85, gamma_R=.90

The first six are genuine interaction contrasts. The last three are signed
regime-boundary shifts included to interpret the nonlinear regions already
revealed by D046--D048.

Every estimand is evaluated for every retained numeric outcome. Scientific
interpretation gives priority to the six core mechanism-to-market outcomes
listed above. Because D049 is exploratory/sequential, interval exclusion of
zero is described as a 95% matched-block interval result rather than as a new
confirmatory significance claim.

## Interpretation guard

No monotone response is assumed in any factor. In particular:

- high alpha can enter a qualitatively different boundary regime;
- high beta can approach an attention-selectivity plateau;
- high gamma_R can reduce raw reputation dispersion relative to sigma_0 and
  thereby attenuate effective selectivity.

Consequently, a three-way interaction may arise from amplification,
saturation, attenuation, or ranking reversal. Do not retrofit a globally
monotone narrative after inspecting the surface.

## Persistence and partial-result guard

Each checkpoint is one indivisible R/SW/SF triplet for one
`(cell_index, replication_id)` pair and is bound to the complete D049/D043/D044
configuration fingerprint.

Final artifacts are written only when all 50 x 300 = 15000 checkpoints validate.
No partial interaction surface or cell ranking may be inspected or used to
change the frozen cells, interaction definitions, replication count,
calibration, or outcome set.

## Planned artifacts

    results/refined/joint_interaction/joint_records.csv
    results/refined/joint_interaction/joint_metadata.json
    results/refined/joint_interaction/joint_analysis.json
    results/refined/joint_interaction/joint_topology_means.csv
    results/refined/joint_interaction/joint_topology_gaps.csv
    results/refined/joint_interaction/joint_pairwise_contrasts.csv
    results/refined/joint_interaction/joint_interactions.csv

## Execution sequence

1. Pass the full refined test gate.
2. Run a small compute-node smoke covering the alpha-zero control, D043 anchor,
   one interior factorial cell, and one boundary factorial cell.
3. If clean, create a fresh `results/refined/joint_interaction` directory.
4. Submit the resumable 300-task production array.
5. Do not inspect partial outcome surfaces.
6. Verify 15000 checkpoints, all array tasks successful, empty stderr, and one
   production commit.
7. Run the separate finalizer only after the complete design validates.
8. Interpret D049 before introducing agent heterogeneity.
