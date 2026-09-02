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

## Paired design / semantic randomness

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py
    src/experiments/refined/treatments.py

Common within paired replication: shock path, non-network initial state, parameters, horizon, evaluation definitions.

Topology-specific: graph seed, realised G, graph-supported W0.

Generated-treatment alpha=0 topology-null control: VERIFIED.

## Topologies / structural validation

    src/topologies/refined/generators.py
    src/topologies/refined/diagnostics.py
    src/experiments/refined/structural.py
    src/experiments/refined/calibration.py
    src/experiments/refined/structural_io.py
    scripts/run_refined_structural_validation.py

D041 structural validation: COMPLETE and VERIFIED.

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

Key conventions:

    return volatility = sample SD
    belief dispersion = population cross-sectional variance
    MAF and Q_F use signed net flow F_t
    exceedance uses >
    stabilisation admissibility uses <=
    L_stab = 50
    censored runs receive no artificial stabilisation time

## Realised influence / common exposure — Eqs. (251)-(265)

    src/experiments/refined/influence_metrics.py

Implements entropy, effective source count, realised source influence/HHI, realised influence of structural hubs, overlap, and attention mobility.

Structural hubs come from directed in-degree in G, never from W_t.

Neutral fixed-out-degree identity:

    W0 = G/K
    s^I_j,0 = d^in_j/(N K)
    S^I_q,0 = S^G_q

Eqs. (266)-(267) KL-to-transition-prior remain deferred with attention inertia.

Status: VERIFIED.

## D042 market-evaluation calibration method

    src/experiments/refined/market_calibration.py

Frozen method:

    T = 1000
    B = 0
    L = 50
    robustness L = {25,100}
    alpha = 0

    scale sample:
        500 runs
        namespace 2026090201
        c_ret, c_bel, c_F = pooled medians

    threshold sample:
        500 separate runs
        namespace 2026090202
        c_CID = 95th percentile of run-level peak CID
        quantile method = higher

    weights = equal thirds
    component guardrails = inactive
    L_stab = 50

Method status: VERIFIED.

## D043 frozen baseline

    src/experiments/refined/baseline_specification.py
    docs/REFINED_BASELINE.md

Canonical API:

    RefinedBaselineSpecification
    first_refined_baseline_specification()

Frozen design:

    N=100, K=6, T=1000, q=5, p_sw=0.02, a0=1.0

Frozen parameters:

    rho_theta=0.985
    sigma_theta=0.025
    v_bar=0
    psi=1
    sigma_s=0.06
    sigma_b=0.025
    alpha=0.75
    kappa=2.4
    x_bar=5
    chi=0.02
    lambda_price=0.0002
    sigma_p=0.001
    gamma_R=0.9
    beta=1
    sigma_0=0.0005

Frozen initialisation:

    theta_0 ~ stationary AR(1)
    b_i,0 = theta_0
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

W0 remains graph-supported uniform attention.

Status: VERIFIED.

## Calibration smoke infrastructure

    src/experiments/refined/calibration_smoke.py
    scripts/run_refined_no_social_calibration_smoke.py
    tests/test_refined_calibration_smoke.py

Smoke-only seeds:

    2026090204 / 2026090205

Observed successful 3+3 smoke:

    c_ret = 0.003087925449
    c_bel = 0.004183656828
    c_F   = 0.1172234222
    c_CID = 1.713734032

These values are explicitly smoke-only and not final calibration values.

Status: VERIFIED at 568-test checkpoint.

## Shared no-social path constructor

    src/experiments/refined/no_social_calibration_paths.py

Responsibilities:

- replace only alpha by zero;
- derive semantic shock / initial-state / graph seeds;
- generate one canonical valid Random fixed-out-degree support;
- construct W0 and neutral non-network initial state;
- run canonical refined simulator;
- compute rolling CID components.

At alpha=0, adaptive and fixed attention must yield exactly identical return/belief/order-flow/CID-component paths under the same randomness. This exact equivalence is regression-tested. Production calibration uses fixed attention only as a computational shortcut; influence diagnostics are not computed from this shortcut.

## Production D042 calibration — NEW

    src/experiments/refined/market_calibration_run.py
    scripts/run_refined_market_calibration.py
    tests/test_refined_market_calibration_run.py

Production runner properties:

- defaults exactly to D042 500+500 protocol;
- sequential scale then threshold stages;
- per-scale-replication compressed NPZ checkpoints;
- per-threshold-replication JSON peak-CID checkpoints;
- resume after interruption;
- deterministic SHA-256 fingerprint of the full D042 protocol + D043 baseline embedded in every checkpoint;
- threshold checkpoint additionally bound to the realised reference-scale fingerprint;
- exact pooled median reference scales;
- exact `higher` empirical quantile of run-level peak CIDs;
- `reference_scales.json` is stage-complete but not a final calibration artifact;
- `market_evaluation_calibration.json` is created only after all threshold runs complete;
- `threshold_peak_cids.csv` preserves all final run-level peaks;
- production path uses `adaptive_attention=False` at alpha=0 after exact equivalence test;
- CLI detects hostnames containing `login` and refuses the 500+500 run unless deliberately overridden.

New production tests: 21.

Expected next checkpoint:

    589 passed

## Current gate

1. Verify 589 tests on Iridis.
2. Do not launch the production script directly on `loginX...`.
3. Submit/run `scripts/run_refined_market_calibration.py` inside an Iridis compute allocation or batch job.
4. Because the runner checkpoints every replication, an interrupted job can be resumed safely.
5. When 500+500 completes, inspect and freeze numerical `c_ret`, `c_bel`, `c_F`, and `c_CID`.
6. Only then build the paired confirmatory market runner and small paired smoke.
7. Large confirmatory topology Monte Carlo remains prohibited until that paired smoke passes.

Formal stability remains separate: equilibrium X*, complete Jacobian J*, `spr(J*)`, Lyapunov analysis. Spectral radius of row-stochastic W is never the market-stability criterion.
