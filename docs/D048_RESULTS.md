# D048 Gamma_R Sweep Results

Status: **COMPLETE AND FINALIZED**

Date: 2026-09-10

## Execution provenance

Frozen design:

    alpha anchor = 0.85
    beta anchor = 5.0
    gamma_R = (0, .5, .8, .9, .95, .98, .99, .995, .999)
    paired replications per gamma_R = 300
    total simulations = 8100
    bootstrap draws = 5000
    bootstrap seed = 2026091002
    experiment seed = 2026091001

Verified execution:

    refined test gate = 766 passed in 32.42s
    smoke job = 1541812
    smoke gamma_R = {0, .9, .999}
    smoke tasks = 3/3 COMPLETED 0:0
    smoke checkpoints = 3/3
    smoke stderr = empty

    production array = 1542666
    production tasks = 54/54 COMPLETED 0:0
    production checkpoints = 2700/2700
    production completion markers = 54/54
    production stderr = empty
    production commit = ce3630faf86d7f55b3535d2a8b902d5c677a6f0c

    finalizer = 1544241
    finalizer state = COMPLETED 0:0
    finalizer host = ruby047
    finalizer elapsed = 00:00:10
    finalizer stderr = empty
    finalizer commit = ce3630faf86d7f55b3535d2a8b902d5c677a6f0c

Final analysis verified cross-gamma common random numbers over all 300
replications.

## Main scientific result

D048 rejects a globally monotone interpretation of reputation persistence.
Over the broad interior range, increasing `gamma_R` reduces attention turnover
and strengthens topology-dependent common exposure and correlated trading.
At the horizon-scale boundary `gamma_R=.999`, however, raw reputation dispersion
becomes small relative to the fixed `sigma_0` floor and the mechanism partially
reverses.

A concise regime map is:

    gamma_R = 0--.5      fast/noisy updating, high attention turnover
    gamma_R = .8--.95    persistent ranking and growing topology amplification
    gamma_R = .98--.995  strongest concentration/covariance differentiation
    gamma_R = .999       regularisation-dominated attenuation/reversal

D048 is exploratory OAT. These descriptions are regime-mapping statements, not
new confirmatory multiple-testing claims.

## Reputation scale and the sigma_0 channel

The new diagnostic

    mean_raw_local_reputation_std_over_sigma0

falls sharply with persistence. Representative topology means are:

    gamma_R=0:
        R  = 3.29497
        SW = 3.10782
        SF = 3.20717

    gamma_R=.90:
        R  = 2.44499
        SW = 2.38292
        SF = 2.33490

    gamma_R=.98:
        R  = 1.43019
        SW = 1.42959
        SF = 1.37990

    gamma_R=.995:
        R  = .69756
        SW = .71277
        SF = .68040

    gamma_R=.999:
        R  = .21316
        SW = .22319
        SF = .21225

Thus, by `.999`, raw local reputation dispersion is far below the fixed
`sigma_0=5e-4` regularisation scale. The resulting attenuation of standardised
reputation differences is a plausible mechanism for the high-persistence
reversal. This is why D049 keeps sigma_0 fixed rather than silently rescaling it
with gamma_R.

## Attention persistence and concentration

Mean attention mobility declines dramatically with gamma_R:

    gamma_R=0:    about .307--.310
    gamma_R=.90:  about .077--.079
    gamma_R=.98:  about .038
    gamma_R=.995: about .017--.018
    gamma_R=.999: about .0052--.0054

Through `.98`, attention entropy and effective-source counts generally fall,
consistent with more persistent/concentrated attention. At `.999` they reverse
sharply:

    mean attention entropy at .999:
        R=.77055, SW=.76682, SF=.77191

    mean effective sources at .999:
        R=4.18004, SW=4.15618, SF=4.18640

compared with effective-source counts near 2.1--2.2 around `.98`.

Hub influence remains strongly topology differentiated throughout. The D048
mechanism is therefore not well described as a monotone change in hub share
alone. The more informative chain is attention persistence/common exposure ->
action covariance -> aggregate-flow variance.

## Mechanism amplification

The unsigned topology gap in attention overlap rises from about 90.49% at
`gamma_R=0` to about 93--94% around `.98--.995`, then falls to 89.91% at `.999`.

The signed action-covariance topology differentiation follows the same broad
pattern. The absolute max-minus-min topology gap is:

    gamma_R=0      .00033360
    gamma_R=.80    .00041766
    gamma_R=.90    .00046016
    gamma_R=.95    .00054613
    gamma_R=.98    .00059702
    gamma_R=.99    .00060178
    gamma_R=.995   .00058737
    gamma_R=.999   .00034699

At every displayed gamma_R, SF exceeds SW in mean pairwise action covariance,
and the matched-block 95% interval for the SW-SF contrast excludes zero.

Aggregate order-flow variance shows corresponding topology amplification. The
relative topology gap is:

    gamma_R=0      6.1613%
    gamma_R=.80    7.6010%
    gamma_R=.90    8.3892%
    gamma_R=.95   10.0008%
    gamma_R=.98   10.9954%
    gamma_R=.99   11.1246%
    gamma_R=.995  10.9277%
    gamma_R=.999   6.6151%

The high-persistence boundary therefore reverses a substantial fraction of the
interior amplification.

## Individual versus correlated trading variation

The topology gap in the sum of individual action variances falls as the
covariance gap rises through the interior region:

    gamma_R=0      4.4710%
    gamma_R=.90    3.1760%
    gamma_R=.98    1.4802%
    gamma_R=.99    1.2107%
    gamma_R=.995   1.2089%

This supports the interpretation that stronger aggregate-flow differentiation
is driven primarily by covariance/common movement rather than by each agent
becoming individually more variable.

## Market translation

Mean absolute order-flow topology differentiation rises materially through the
interior region and then recedes:

    gamma_R=0      .3005%
    gamma_R=.90   1.6633%
    gamma_R=.95   2.4967%
    gamma_R=.98   3.2137%
    gamma_R=.99   3.3244%
    gamma_R=.995  3.4099%
    gamma_R=.999  1.8855%

Return-volatility differentiation follows the same qualitative pattern:

    gamma_R=0      .4644%
    gamma_R=.90    .8403%
    gamma_R=.95   1.1684%
    gamma_R=.98   1.3572%
    gamma_R=.99   1.3840%
    gamma_R=.995  1.4293%
    gamma_R=.999   .8725%

The level of return volatility remains close to .0033 across the grid. Thus
D048 mainly changes cross-topology differentiation rather than generating a
large increase in the overall volatility level.

Peak CID is small and non-monotone across topology contrasts. Most importantly,
`threshold_exceeding=0` for R, SW, and SF at every gamma_R. D048 therefore does
not generate severe threshold-level operational instability under the frozen
calibration.

Mispricing topology gaps generally shrink with persistence and are tiny near the
upper range. D048 should not be interpreted as a broad monotone worsening of
all market-quality outcomes.

## Mechanism statement for the report

Preferred concise interpretation:

> Reputation persistence amplifies topology-dependent market propagation over a
> broad interior range by reducing attention turnover and increasing common
> exposure and correlated trading. This amplification is strongest at high but
> non-boundary persistence. At gamma_R=.999, however, reputation dispersion
> becomes small relative to the fixed regularisation floor, attenuating
> effective reputational selectivity and partially reversing the propagation
> mechanism.

The evidence supports the chain

    gamma_R up (interior range)
      -> attention turnover down
      -> common exposure / overlap up
      -> action covariance differentiation up
      -> aggregate-flow variance differentiation up
      -> order-flow and return-volatility differentiation up

with a boundary attenuation channel at `.999` through the fixed sigma_0 floor.

## Implication for D049

D048 makes a joint experiment scientifically necessary: persistence and
selectivity are not practically separable when the reputation standardisation
floor is fixed. D049 therefore varies alpha, beta, and gamma_R jointly while
keeping sigma_0 fixed and analyses signed topology interactions rather than
continuing OAT sweeps.
