# Refined Model Implementation Map

Last updated: 2026-09-02

Scientific source of truth: `report1_25_08_2026.pdf`. Legacy code is reference only.

## Core runtime — Eqs. (35)-(82)

    src/model/refined/state.py
    src/model/refined/shocks.py
    src/model/refined/fundamentals.py
    src/model/refined/beliefs.py
    src/model/refined/trading.py
    src/model/refined/market.py
    src/model/refined/reputation.py
    src/model/refined/attention.py
    src/model/refined/transition.py
    src/model/refined/simulator.py

Binding transition:

    W_{t-1}
      -> theta_t, v_t, s_t
      -> b_t
      -> vhat_t -> m_t -> desired action -> executed action
      -> x_t -> F_t -> p_t -> r_t -> pi_t -> R_t -> z_t -> W_t

Status: VERIFIED.

## Paired design and semantic randomness

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py
    src/experiments/refined/treatments.py

Common within paired replication: shock path, non-network initial state, parameters, horizon, evaluation definitions.

Topology-specific: graph seed, realised G, graph-supported W0.

Generated-treatment alpha=0 topology-null control: VERIFIED.

## Topology generation and structural validation

    src/topologies/refined/generators.py
    src/topologies/refined/diagnostics.py
    src/experiments/refined/structural.py
    src/experiments/refined/calibration.py
    src/experiments/refined/structural_io.py
    scripts/run_refined_structural_validation.py

Report mapping:

    Eqs. (183)-(212): topology definitions / relabelling
    Eqs. (203)-(211): structural diagnostics

D041 structural run: COMPLETE and VERIFIED.

## Market outcomes and CID

    src/experiments/refined/market_metrics.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/cid.py
    src/experiments/refined/cid_events.py

Mapping:

    Eqs. (231)-(235): evaluation sample / full rolling windows
    Eqs. (236)-(238): RV, RMSM, MAM, MAF
    Eqs. (239)-(240): rolling pairwise action covariance + exact Var(F) decomposition
    Eqs. (241)-(246): CID raw components, reference scales, weights, CID
    Eqs. (247)-(250): exceedance, duration, operational stabilisation, censoring
    Eqs. (288)-(289): MAR and time-averaged cross-sectional belief variance

Status: VERIFIED.

Important conventions:

    return volatility uses sample SD
    belief dispersion uses population cross-sectional variance
    MAF and Q_F use signed net flow F_t, not gross volume
    exceedance uses >
    stabilisation admissibility uses <=
    L_stab = 50
    censored runs receive no artificial stabilisation time

## Realised influence and common exposure — Eqs. (251)-(265)

    src/experiments/refined/influence_metrics.py

Implements normalised attention entropy, effective source count, realised source influence shares/HHI, realised influence of structural hubs, overlap, and RMS attention mobility.

Structural hubs are selected from directed in-degree in G, never from W_t.

Neutral fixed-out-degree identity enforced:

    W0 = G/K
    s^I_j,0 = d^in_j/(N K)
    S^I_q,0 = S^G_q

Eqs. (266)-(267) KL-to-transition-prior remain deferred with attention inertia.

Status: VERIFIED.

## Market-evaluation calibration method — D042

    src/experiments/refined/market_calibration.py

Method:

    T = 1000
    B = 0
    L = 50
    robustness L = {25,100}
    alpha_calibration = 0

    scale sample:
        500 runs, namespace 2026090201
        c_ret, c_bel, c_F = pooled medians of raw rolling components

    threshold sample:
        500 separate runs, namespace 2026090202
        c_CID = 95th percentile of run-level peak CID
        quantile convention = higher

    weights = equal thirds
    baseline component guardrails = inactive
    L_stab = 50

Status: VERIFIED at 445-test checkpoint.

This freezes the METHOD only. Numerical scales and c_CID must not be produced until the maintained refined market specification is fixed.

## Provisional refined baseline specification

    src/experiments/refined/baseline_specification.py
    docs/REFINED_BASELINE_CANDIDATE.md

Status: PROVISIONAL, NOT YET FROZEN.

Candidate dimensions:

    N = 100
    K = 6
    T = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

Candidate parameters:

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

Provisional non-network initialisation:

    theta_0 ~ stationary N(0, sigma_theta^2/(1-rho_theta^2))
    b_i,0 = theta_0
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

W0 remains graph-supported uniform attention.

## Pre-freeze baseline scale smoke — VERIFIED AND RUN

    src/experiments/refined/market_smoke.py
    scripts/run_refined_baseline_scale_smoke.py
    tests/test_refined_market_smoke.py

Verified at the 504-test checkpoint and executed on Iridis with five paired replications / 15 treatment runs.

Key pooled medians:

    return_std                              0.00342286
    rms_mispricing                          0.0685422
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

Interpretation: return/mispricing/flow scales are finite and non-degenerate, actions are not tanh-saturated, and inventory bounds are active but not mechanically dominant. The unresolved pre-freeze parameter is sigma_0 because the provisional floor is hundreds of times smaller than typical local reputation dispersion.

## sigma_0 common-random-number sensitivity gate — NEW

    src/experiments/refined/sigma0_sensitivity.py
    scripts/run_refined_sigma0_sensitivity_smoke.py
    tests/test_refined_sigma0_sensitivity.py

Purpose: evaluate only the regularisation scale of Eq. (58) before freezing the baseline.

Controlled OAT grid:

    sigma_0 in {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

Design:

    experiment_seed = 2026090203
    paired replications = 5
    same replication ids for every sigma_0
    graph/shock/initial-state randomness common across sigma_0 values
    only sigma_0 changes

Output is pooled across topology labels. No topology contrasts/rankings are computed. Reported metrics include reputation dispersion relative to sigma_0, attention mobility/distance, return scale, mispricing, signed flow and inventory projection.

New test file contributes 24 cases.

Expected checkpoint:

    528 tests

## Current gate before calibration / confirmatory Monte Carlo

Required order:

1. verify 528-test checkpoint;
2. run `python scripts/run_refined_sigma0_sensitivity_smoke.py`;
3. select sigma_0 on regularisation/scale grounds only;
4. document and freeze the complete baseline vector if the selected specification remains non-degenerate;
5. recheck alpha=0 topology-null property under the frozen vector;
6. run small no-social D042 calibration smoke;
7. run full D042 500+500 calibration and persist c_ret, c_bel, c_F, c_CID;
8. build paired market-output persistence layer and small paired smoke;
9. only then submit large confirmatory topology Monte Carlo.

Formal stability remains separate: equilibrium X*, complete Jacobian J*, spr(J*), Lyapunov analysis. The spectral radius of row-stochastic W is never the market-stability criterion.
