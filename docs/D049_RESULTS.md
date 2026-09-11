# D049 Joint alpha-beta-gamma_R Interaction Results

Status: **COMPLETE, FINALIZED, AND INTERPRETED**

Date: 2026-09-11

## Execution provenance

Frozen design:

    alpha factorial levels = (.40, .85, .99)
    beta factorial levels = (0, 1, 5, 100)
    gamma_R factorial levels = (0, .90, .99, .999)
    factorial cells = 48
    alpha-zero control = (0, 5, .90)
    D043 anchor = (.75, 1, .90)
    total cells = 50
    paired replications per cell = 300
    topology triplet = (R, SW, SF)
    checkpoints = 15000
    simulations = 45000
    bootstrap draws = 5000
    experiment seed = 2026091003
    bootstrap seed = 2026091004
    sigma_0 = .0005 fixed

Verified code and execution:

    refined test gate = 800 passed in 53.24s
    pre-production clean HEAD = fb4ca7f4d4c5b9ad9fa93d61c3cfbd0153b0bf6a

    smoke job = 1544858
    smoke cells = F26 (.85,5,.99), F47 (.99,100,.999), C_ALPHA0, A_D043
    smoke tasks = 4/4 COMPLETED 0:0
    smoke checkpoints = 4/4
    smoke completion markers = 4/4
    smoke stderr = empty

    production array = 1545569
    production tasks = 300/300 completed successfully
    production checkpoints = 15000/15000
    production completion markers = 300/300
    production stderr = empty
    production commit = fb4ca7f4d4c5b9ad9fa93d61c3cfbd0153b0bf6a

    finalizer = 1547957
    finalizer state = COMPLETED 0:0
    finalizer host = ruby047
    finalizer elapsed = 00:00:31
    finalizer stderr = empty
    finalizer commit = fb4ca7f4d4c5b9ad9fa93d61c3cfbd0153b0bf6a

Final analysis verifies both:

    cross-cell common random numbers = True
    alpha=0 exact economic-path topology null = True

D049 is a sequential exploratory interaction experiment. Factor levels were
selected after D046--D048. No new Holm/FWER family is claimed. Evidence below is
reported using matched complete-block 95% percentile bootstrap intervals.

## Main scientific result

The central D049 result is that topology-dependent propagation is conditional
and regime dependent. Network architecture does not have a single invariant
market effect. Instead, topology interacts with the strength of social
transmission (`alpha`), reputational selectivity (`beta`), and reputation
persistence (`gamma_R`).

A concise statement is:

> Topology matters conditionally: instability emerges when network architecture
> interacts with sufficiently strong social transmission, selective attention,
> and reputation persistence.

The broad interior mechanism is:

    stronger social transmission / selective attention / persistent reputation
      -> more common exposure
      -> larger pairwise action covariance
      -> larger aggregate order-flow variance
      -> greater order-flow and return-volatility differentiation

This amplification is nonlinear. High-persistence regularisation attenuation
and the high-alpha boundary can weaken or reverse the interior ranking.

## Signed topology estimand

The primary D049 interaction quantity is

    D(alpha,beta,gamma_R;Y) = Y_SF - Y_SW.

Positive D means the scale-free/hub-dominated topology exceeds the small-world
topology for outcome Y. Interaction contrasts are finite differences of D, not
unsigned max-minus-min topology gaps.

## Interior beta x gamma_R complementarity

At `alpha=.85`, the predeclared
`beta_gamma_core_at_alpha085` contrast compares the beta response from `0 -> 5`
across the gamma_R response from `0 -> .99`.

For every priority outcome, the matched-block 95% interval excludes zero:

    attention overlap                  +0.00203520  [0.00135764, 0.00270301]
    pairwise action covariance         +0.00026402  [0.00020793, 0.00032035]
    aggregate order-flow variance      +2.6492564   [2.0885833, 3.2129691]
    mean absolute order flow           +0.00206472  [0.00174632, 0.00239206]
    return volatility                  +3.02851e-05 [2.34957e-05, 3.70247e-05]
    peak CID                           +0.00907836  [0.00060034, 0.01758269]

Thus beta and gamma_R are complementary over the interior range: persistent
reputation makes selective attention more consequential for topology-dependent
propagation.

The central `alpha=.85` surface makes this visible. At `beta=5`, moving from
`gamma_R=0` to `.99` changes SF-SW differentiation as follows:

    pairwise action covariance      .00029828 -> .00056230
    aggregate flow variance         2.89645    -> 5.54571
    mean absolute order flow       -.0000223   -> .00204244
    return volatility              1.2129e-05 -> 4.2414e-05

The mechanism is therefore not a simple level effect of one behavioural
parameter. Persistence changes how selectivity maps the fixed network
architecture into common trading behaviour.

## Alpha interactions in the interior

At `gamma_R=.90`, the predeclared `alpha_beta_core_at_gamma09` interaction is
positive for attention overlap, action covariance, aggregate flow variance,
mean absolute order flow, and return volatility. The 95% matched-block interval
excludes zero for each of those outcomes. Peak CID does not have a corresponding
interval exclusion.

Likewise, at `beta=5`, the predeclared `alpha_gamma_core_at_beta5` interaction is
positive for action covariance, aggregate-flow variance, mean absolute order
flow, return volatility, and peak CID. Attention-overlap interaction is small and
its interval includes zero.

This implies that stronger social transmission increases the downstream market
consequence of persistence/selectivity interactions even when the upstream
attention-overlap interaction alone is not clearly separated from zero.

## Three-way alpha x beta x gamma_R interaction

The predeclared three-way contrast

    alpha_beta_gamma_core

has a 95% matched-block interval excluding zero for:

    pairwise action covariance
    aggregate order-flow variance
    mean absolute order flow
    return volatility
    peak CID

The interval for attention overlap includes zero.

Numerically, for the priority downstream outcomes:

    pairwise action covariance       +0.00030454 [0.00024710, 0.00036099]
    aggregate order-flow variance    +3.052575   [2.476915, 3.616102]
    mean absolute order flow         +0.00219901 [0.00187562, 0.00252669]
    return volatility                +3.31073e-05 [2.63482e-05, 3.98406e-05]
    peak CID                         +0.01020433 [0.00185521, 0.01879523]

The equality of the reported `alpha_gamma_core_at_beta5` and three-way numbers
is structural rather than a coding error. At `beta=0`, reputation is irrelevant
to attention weights, so the gamma_R difference in the topology contrast is
exactly zero. The beta=0 side of the three-way finite difference therefore
cancels.

## High-persistence boundary and sigma_0 attenuation

D048 found that `gamma_R=.999` pushes raw reputation dispersion below the fixed
regularisation scale. D049 reproduces that mechanism inside the joint design.
At `alpha=.85, beta=5`, the pooled topology mean of
`mean_raw_local_reputation_std_over_sigma0` is approximately:

    gamma_R=0      3.15733
    gamma_R=.90    2.35948
    gamma_R=.99    1.01725
    gamma_R=.999    .21931

At the same cells:

    attention mobility     .30759 -> .07825 -> .02645 -> .00520
    attention entropy      .41031 -> .38589 -> .38935 -> .76539
    effective sources      2.5593 -> 2.3540 -> 2.2480 -> 4.1444
    influence HHI          .03749 -> .03857 -> .03910 -> .02398
    attention overlap      .03158 -> .03250 -> .03311 -> .02088

Thus `.999` is not simply "more persistence" along the same interior path.
Raw reputation dispersion becomes much smaller than sigma_0, standardised
reputation differences are attenuated, and attention becomes more diffuse.

The predeclared `gamma_boundary_shift_at_alpha085_beta5` confirms the reversal
in signed topology propagation from `.99 -> .999`:

    attention overlap                  -0.01234967
    pairwise action covariance         -0.00023266
    aggregate order-flow variance      -2.3039155
    mean absolute order flow           -0.00081730
    return volatility                  -1.44904e-05

Each corresponding 95% matched-block interval excludes zero. Peak CID does not.

## Beta can offset boundary attenuation

The high-persistence attenuation is itself beta dependent. At `alpha=.85`, the
central surface for aggregate-flow variance is:

                  gamma_R
    beta       0       .90      .99      .999
      0      2.152    2.152    2.152    2.152
      1      1.511    1.895    2.308    2.191
      5      2.896    4.577    5.546    3.242
    100      4.601    6.352    8.854    8.920

For return volatility, the signed SF-SW surface is:

                  gamma_R
    beta       0          .90        .99        .999
      0     1.88e-05   1.88e-05   1.88e-05   1.88e-05
      1     1.02e-05   1.54e-05   1.99e-05   1.93e-05
      5     1.21e-05   3.00e-05   4.24e-05   2.79e-05
    100     1.84e-05   3.95e-05   6.23e-05   6.40e-05

Thus the `.999` attenuation is strong at beta=5 but largely offset at beta=100.
The boundary `beta_gamma` interaction is positive for overlap, covariance,
aggregate-flow variance, mean absolute order flow, and return volatility, with
matched-block 95% intervals excluding zero.

This is consistent with the interpretation that a sufficiently large beta can
partly compensate for smaller standardised reputation differences. It does not
remove the sigma_0 mechanism; it changes its effective strength.

## High-alpha boundary regime and topology-ranking reversal

D046 already indicated that the upper alpha region should not be described as a
smooth monotone continuation. D049 confirms this in the joint design.

At `beta=5, gamma_R=.90`, the signed boundary shift from
`alpha=.85 -> .99` is:

    attention overlap                  -0.00760681
    pairwise action covariance         -0.00028227
    aggregate order-flow variance      -3.1452151
    mean absolute order flow           -0.00359063
    return volatility                  -0.00018176
    peak CID                           -0.22036229

Every corresponding 95% matched-block interval excludes zero.

For return volatility, D at `alpha=.85` is approximately `+2.998e-05`. Applying
the boundary shift implies a negative D at `.99`, so the SF-SW ordering reverses.
The high-alpha region is therefore a qualitatively different regime, not merely
an attenuated version of the interior amplification regime.

## Exact alpha-zero negative control

The alpha-zero control is exact. At `(alpha,beta,gamma_R)=(0,5,.90)`, R, SW,
and SF have identical economic paths for every matched replication.
Representative final means are exactly equal across topologies:

    aggregate order-flow variance = 113.17023
    mean absolute order flow      = .092791867
    return volatility             = .0038232022
    peak CID                      = 1.5539693
    RMS mispricing                = .066029951

Therefore topology differences in D045--D049 operate through the social-network
channel rather than through an uncontrolled topology-correlated economic shock.

## D043 continuity anchor

At the explicit D043 anchor `(.75,1,.90)`, the final D049 means are close to the
previous frozen benchmark and preserve the familiar modest market
cross-topology differentiation:

    aggregate-flow variance D(SF-SW) = 1.610604
    mean absolute order flow D        = .0003744519
    return-volatility D               = 1.19119e-05
    peak-CID D                        = .00192621
    RMS-mispricing D                  = -9.20e-06

This provides continuity between the original homogeneous benchmark and the
joint interaction surface.

## Interpretation guard

D049 does not support a globally monotone story in alpha, beta, or gamma_R.
The relevant regimes are conditional:

    interior amplification:
        moderate/high alpha + selective beta + persistent gamma_R
        -> stronger topology-dependent common trading and market propagation

    gamma_R boundary:
        very persistent reputation + fixed sigma_0
        -> compressed standardised reputation differences
        -> partial attenuation unless beta is sufficiently large

    alpha boundary:
        alpha near one
        -> qualitatively different transmission regime
        -> possible topology-ranking reversal in market outcomes

No partial or boundary pattern should be described as a universal comparative
static.

## Preferred report statement

> Network architecture has a conditional rather than invariant effect on price
> instability. In the interior regime, stronger social transmission,
> reputational selectivity, and reputation persistence interact to increase
> common exposure, correlated trading, and aggregate order-flow variance, which
> strengthens topology-dependent return-volatility differences. At extreme
> persistence, the fixed reputation-standardisation floor attenuates effective
> selectivity, while near-unit social weight can move the system into a distinct
> boundary regime in which topology rankings may reverse.

## Implication for the thesis structure

D045--D049 should no longer be narrated as five isolated experiments. Together
they form one identification sequence:

    D045  topology alone under the frozen benchmark
      -> strong mechanism differences, modest realised market differences

    D046  vary social transmission alpha
      -> identify interior amplification and high-alpha boundary behaviour

    D047  vary reputational selectivity beta
      -> show that selectivity amplifies rather than activates topology effects

    D048  vary reputation persistence gamma_R
      -> identify persistence amplification and sigma_0 boundary attenuation

    D049  vary alpha, beta, gamma_R jointly
      -> establish interaction and regime dependence

This supports a substantial rewrite of Section 6 around the single thesis-level
message that topology matters conditionally. Heterogeneity should be introduced
only after this homogeneous interaction map has been documented and visualised.
