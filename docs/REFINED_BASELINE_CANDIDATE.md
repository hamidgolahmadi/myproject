# Provisional Refined Baseline Candidate

Status: **PROVISIONAL — NOT YET FROZEN**

This note records the first numerical candidate for the refined homogeneous
market model. It is not a report equation and must pass pre-freeze scale checks
before becoming a binding design decision.

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
    sigma_0      = 1e-6   # under explicit pre-freeze review

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
bound from a cumulative inventory bound.

`sigma_0` is the positive floor in the regularised local reputation dispersion
in Equation (58). The report states that the floor prevents an almost
degenerate local reputation distribution from producing an artificial
numerical explosion after standardisation. The original provisional value
`1e-6` was chosen only as a minimal numerical floor and is therefore subject to
an explicit scale sensitivity check before freeze.

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

## First paired scale smoke — completed 2026-09-02

Design:

    experiment_seed = 2026090203
    paired replications = 5
    topology treatments per replication = 3
    total runs = 15
    N = 100
    T = 1000

The smoke pooled all treatment-labelled runs and was used only for absolute
scale/non-degeneracy assessment, not topology ranking.

Pooled medians across 15 runs:

    return_std                              0.00342286
    mean_abs_return                         0.00271065
    max_abs_return                          0.0109005
    rms_mispricing                          0.0685422
    max_abs_mispricing                      0.219996
    mean_abs_flow_per_agent                 0.0784241
    rms_flow_per_agent                      0.101498
    desired_action_abs_p95                  0.304871
    desired_action_saturation_fraction      0.0
    execution_projection_fraction           0.14608
    inventory_boundary_fraction             0.14608
    median_local_reputation_std              0.000672808
    median_reputation_scale_to_sigma0        672.808
    mean_attention_mobility                  0.0492363
    max_attention_mobility                   0.664115
    final_attention_distance_from_initial    0.379522

Interpretation before any topology analysis:

- return, mispricing, and signed order-flow variation are non-zero and finite;
- desired actions are well away from tanh saturation;
- the inventory bound is economically active but not mechanically binding in
  every period;
- the main unresolved scale issue is `sigma_0`: typical local reputation
  dispersion is hundreds of times larger than the provisional floor, so the
  floor becomes negligible almost immediately once reputation differences
  appear.

Therefore the rest of the parameter vector is not currently flagged for
revision, but the baseline cannot yet be frozen because `sigma_0=1e-6` needs a
controlled OAT scale check.

## sigma_0 sensitivity gate

A common-random-number OAT smoke is pre-specified over:

    sigma_0 in {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

All graph, shock, and initial-state randomness is reused across candidate
values. The output is pooled across topology labels and reports only absolute
scale diagnostics. It must not be used to tune the model toward a preferred
R/SW/SF ranking.

The purpose is to determine whether the reputation floor is functionally
negligible, excessively damped, or operating on a scale comparable with the
realised local reputation dispersion. Market return/mispricing/flow diagnostics
are reported alongside attention mobility to ensure that changing the floor
does not create a new degeneracy elsewhere.

## Remaining gate before freeze

Do not freeze this candidate or run the D042 500+500 calibration until:

1. the sigma_0 sensitivity smoke is completed and interpreted;
2. one value is selected on scale/regularisation grounds only;
3. the selected complete baseline vector is recorded as a frozen decision;
4. the alpha=0 topology-null property is rechecked under that frozen vector.

The smoke stage is a scale/non-degeneracy check, not a search for a topology
ranking. If the candidate is revised, the reason must be documented before any
CID calibration sample is generated.
