# Frozen First Refined Baseline

Status: **FROZEN — D043**

This document records the maintained first-stage homogeneous refined market
specification used by the D042 no-social calibration and the subsequent first
confirmatory fixed-topology market experiments.

The doctoral report remains the scientific source of truth for equations and
timing.  The numerical values below are an explicit experiment design because
the report does not provide one complete numerical parameter table for the
refined model.

## Dimensions and topology treatment settings

    N = 100
    K = 6
    T = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

The topology settings are inherited from the structurally validated D041
configuration.

## Frozen homogeneous RefinedParameters

    rho_theta    = 0.985
    sigma_theta  = 0.025
    v_bar        = 0.0
    psi          = 1.0
    sigma_s      = 0.06
    sigma_b      = 0.025
    alpha        = 0.75
    kappa        = 2.4
    x_bar        = 5.0
    chi          = 0.02
    lambda_price = 0.0002
    sigma_p      = 0.001
    gamma_R      = 0.9
    beta         = 1.0
    sigma_0      = 0.0005

## Frozen neutral non-network initialisation

For each replication, using the semantic `initial_state_seed`:

    theta_0 ~ N(0, sigma_theta^2/(1-rho_theta^2))
    b_i,0 = theta_0 for every i
    p_0 = v_bar + psi * theta_0
    x_i,0 = 0
    R_i,0 = 0

`W_0(G)` is constructed separately and uniformly over each realised graph's
feasible neighbours under D018/D025.

The purpose of `b_0=theta_0 1` and `p_0=v_0` is neutral initialisation, not a
claim that real traders observe the latent state.  Because the first-stage
evaluation uses `B=0`, arbitrary initial mispricing or arbitrary disagreement
would otherwise mechanically enter the measured sample.  Private signal and
belief noise enter normally from period 1.

## Numerical provenance

The information-process parameters and behavioural anchors use the historical
pilot only where roles and units remain comparable.  Legacy code never overrides
the refined equations.

The trading slope `kappa=2.4` matches the historical effective local tanh slope
rather than copying a legacy raw coefficient whose gate differed.

The pilot price process used a level price near 100 and simple returns.  The
refined model uses normalised/log-price changes, so the provisional price-impact
and price-noise anchors were divided by 100:

    0.02 / 100 -> lambda_price = 0.0002
    0.10 / 100 -> sigma_p      = 0.001

`chi=0.02` and `x_bar=5.0` are refined design choices because the refined model
contains an explicit fundamental anchor and a separate cumulative inventory
constraint.

## Pre-freeze scale smoke

A topology-blind pooled smoke used 5 paired replications, 3 topology treatments
per replication, N=100, T=1000, and experiment namespace `2026090203`.

The initial candidate with `sigma_0=1e-6` showed:

    median return SD                         0.00342286
    median RMS mispricing                    0.0685422
    median RMS signed flow per agent         0.101498
    median |desired action| p95              0.304871
    desired-action saturation fraction      0
    median execution projection fraction     0.14608
    median inventory-boundary fraction       0.14608

These values established finite, non-degenerate market dynamics without tanh
saturation or permanent inventory domination.

The same smoke also showed that the original `sigma_0=1e-6` was negligible
relative to realised local reputation dispersion, motivating a dedicated OAT
regularisation-floor sensitivity before freezing the baseline.

## sigma_0 sensitivity and final choice

A common-random-number OAT sensitivity used the same 5 paired replications,
graphs, shock paths, and initial states for:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}.

Across this grid, return volatility, mispricing, order flow, desired-action
scale, and inventory projection were essentially unchanged.  The main effect
was the intended one: larger `sigma_0` moderated reputation-score
standardisation and therefore reduced attention mobility.

At `sigma_0=5e-4`, pooled medians were approximately:

    median raw local reputation std / sigma_0    1.357
    mean attention mobility                       0.02950
    max attention mobility                        0.32234
    final W distance from W_0                     0.29332
    return SD                                     0.003428
    RMS mispricing                                0.068518
    execution projection fraction                 0.14385

This value is frozen because the regularisation floor is of the same order as
realised local reputation dispersion: it is large enough to matter near a
nearly degenerate reputation distribution, as intended by Equation (58), but it
does not dominate realised dispersion or suppress adaptive attention.  The
choice was made from pooled absolute diagnostics and not from topology rankings
or confirmatory treatment effects.

## Calibration gate

With D043 frozen, the maintained market specification is now available for the
D042 no-social calibration protocol.  The next allowed step is a small
end-to-end no-social calibration smoke.  Only after that smoke succeeds may the
full 500 scale + 500 threshold calibration be run and the numerical
`c_ret`, `c_bel`, `c_F`, and `c_CID` be persisted and frozen.
