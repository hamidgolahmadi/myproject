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

D041 structural run: COMPLETE and VERIFIED. R/SW/SF are strongly separated in intended structural dimensions.

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
    L_stab = 50 in first-stage report protocol
    no artificial stabilisation time for censored runs

## Realised influence and common exposure — Eqs. (251)-(265)

    src/experiments/refined/influence_metrics.py

Implements:

    normalised attention entropy
    effective number of sources
    realised source influence shares
    realised-influence HHI
    realised influence share of structural hubs H_q(G)
    attention overlap and equivalent matrix identity
    RMS attention mobility

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

## Provisional refined baseline specification — NEW

    src/experiments/refined/baseline_specification.py
    docs/REFINED_BASELINE_CANDIDATE.md

Status: PROVISIONAL, AWAITING IRIDIS TEST + SCALE SMOKE.

Candidate dimensions:

    N = 100
    K = 6
    T = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

Candidate `RefinedParameters`:

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

Provisional non-network initialisation:

    theta_0 ~ stationary N(0, sigma_theta^2/(1-rho_theta^2))
    b_i,0 = theta_0
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

`W_0` remains generated separately from G using the frozen uniform-support rule.

The candidate uses pilot values only as provenance where units/roles remain comparable. Pilot level-price coefficients are converted to refined normalised return units before use as candidate anchors. `chi`, `x_bar`, and `sigma_0` are genuinely new refined choices and must be assessed by smoke diagnostics.

New tests:

    tests/test_refined_baseline_specification.py

Expected next checkpoint:

    474 tests

## Current gate before calibration / confirmatory Monte Carlo

Do not run D042 500+500 calibration yet.

Required order:

1. verify 474-test checkpoint;
2. build small paired scale/non-degeneracy smoke runner for the provisional baseline;
3. inspect scale only, not topology rankings;
4. verify returns/mispricing/net flow/action saturation/inventory contacts/reputation scale/influence diagnostics/finiteness;
5. if non-degenerate, promote candidate to frozen baseline decision;
6. run small no-social calibration smoke;
7. run full D042 500+500 calibration and persist c_ret, c_bel, c_F, c_CID;
8. build paired market-output persistence layer and small paired smoke;
9. only then submit large confirmatory topology Monte Carlo.

Formal stability remains a later and separate task: equilibrium X*, complete Jacobian J*, spr(J*), Lyapunov analysis. The spectral radius of row-stochastic W is never the market-stability criterion.
