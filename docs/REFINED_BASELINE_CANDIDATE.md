# Provisional Refined Baseline Candidate

Status: **PROVISIONAL — NOT YET FROZEN**

This note records the first numerical candidate for the refined homogeneous
market model. It is not a report equation and must pass a scale/non-degeneracy
smoke experiment before becoming a binding design decision.

## Candidate dimensions

    N = 100
    K = 6
    T = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

The topology settings are inherited from the already validated D041 structural
configuration.

## Candidate RefinedParameters

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
    sigma_0      = 1e-6

## Provenance

The information-process values `rho_theta`, `sigma_theta`, `sigma_s`,
`sigma_b`, `alpha`, `gamma_R`, and `beta` use the legacy pilot only as a
numerical anchor because their economic roles remain close to the refined
specification. They are not treated as report-mandated numbers.

The candidate local trading slope is `kappa=2.4`. In the historical fixed
baseline, the effective tanh slope was `trade_sensitivity * gate = 6 * 0.4 =
2.4`; the historical adaptive environment also used `2.4` directly.

The pilot price process used a level price near 100 and simple returns. The
refined model uses price changes in normalised/log-price units. The provisional
mapping therefore divides the pilot level-price coefficients by the reference
price 100:

    legacy price_impact 0.02 / 100 -> lambda_price 0.0002
    legacy sigma_price  0.10 / 100 -> sigma_p      0.001

This is a unit-conversion anchor, not empirical calibration.

`chi=0.02` is new because the refined price equation contains an explicit
fundamental anchor absent from the historical pilot price law.

`x_bar=5.0` is new because the refined model separates the one-period tanh
bound from a cumulative inventory bound. It is deliberately loose enough that
inventory constraints should not dominate every period, but this must be
checked rather than assumed.

`sigma_0=1e-6` is a small positive reputation-dispersion floor in the units of
reputation. It prevents a zero denominator at initially equal reputations while
becoming negligible once economically meaningful local reputation dispersion
appears. Its adequacy must be checked in the scale smoke.

## Provisional neutral initialisation

For each replication, using the semantic `initial_state_seed`:

1. Draw `theta_0` from the stationary AR(1) distribution:

       theta_0 ~ N(0, sigma_theta^2 / (1-rho_theta^2)).

2. Set all initial state beliefs equal to that state:

       b_i,0 = theta_0.

3. Start price exactly at contemporaneous fundamental value:

       p_0 = v_bar + psi * theta_0.

4. Set:

       x_0 = 0
       R_0 = 0.

5. Construct `W_0(G)` separately and uniformly over feasible neighbours, as
   already frozen in D018/D025.

The purpose of `b_0=theta_0 1` and `p_0=v_0` is neutrality, not a claim that
real investors know the true latent state. Because the first-stage evaluation
uses `B=0`, starting from arbitrary price mispricing or arbitrary belief
dispersion would mechanically contaminate the measured early path. Private
signal and belief noise begin from period 1 under the normal transition law.

## Required smoke checks before freeze

Do not freeze this candidate or run the D042 500+500 calibration until a small
paired market smoke experiment reports, without tuning to topology rankings:

- return and mispricing distributions;
- signed net-flow per agent;
- desired-action magnitude / tanh saturation;
- inventory-bound contact frequency;
- reputation magnitude relative to `sigma_0`;
- realised-influence HHI / overlap / mobility;
- numerical finiteness for all paths;
- the alpha=0 topology-null property under the same candidate.

The smoke stage is a scale/non-degeneracy check, not a search for a topology
ranking. If the candidate is revised, the reason must be documented before any
CID calibration sample is generated.
