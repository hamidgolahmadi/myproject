# Refined Model Implementation Map

Last updated: 2026-09-02

## 1. Source of truth

Scientific source of truth:

    report1_25_08_2026.pdf

If this map conflicts with the report, the report wins. Legacy code is reference/reproducibility only.

---

## 2. Architecture

Core economic runtime:

    src/model/refined/

Topology layer:

    src/topologies/refined/
        generators.py
        diagnostics.py

Experiment/evaluation layer:

    src/experiments/refined/
        seeding.py
        paired.py
        treatments.py
        structural.py
        calibration.py
        structural_io.py
        market_metrics.py
        action_covariance.py
        cid.py
        cid_events.py
        influence_metrics.py
        market_calibration.py

Rule: economic transition equations live only under `src/model/refined/`. Evaluation and calibration modules consume completed outputs and must not duplicate or alter the transition law.

---

## 3. Core runtime, Eqs. (35)-(82)

Persistent state:

    X_t = (theta_t, b_t, x_t, p_t, R_t, W_t)

Binding Eq. (39) coordinator:

    src/model/refined/transition.py
    transition_one_period(...)

Critical timing:

    b_t uses W_{t-1}; W_t first affects b_{t+1}

Key mapping:

    Eqs. (35)-(41) state.py, attention.py
    Eqs. (42)-(47) fundamentals.py, shocks.py
    Eqs. (48)-(55) beliefs.py
    Eqs. (56)-(62) attention.py
    Eqs. (63)-(73) trading.py
    Eqs. (74)-(79) market.py, reputation.py
    Eqs. (80)-(82) transition.py

Multi-period wrapper:

    src/model/refined/simulator.py
    SimulationResult
    simulate_shock_path(...)

Status: VERIFIED.

---

## 4. Randomness and paired topology design

Semantic seed architecture:

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py

Roles include:

    replication_id
    graph_seed
    shock_seed
    initial_state_seed
    type_assignment_seed

Paired treatment preparation:

    src/experiments/refined/treatments.py
    prepare_paired_treatments(...)

Common within replication: shocks, non-network initial conditions, parameters, horizon, evaluation definitions.

Topology-specific: graph seed/realisation and graph-supported `W_0`.

Generated-treatment `alpha=0` topology-null control: VERIFIED.

---

## 5. Benchmark graph generation and structural validation

Generators:

    src/topologies/refined/generators.py
    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

Structural diagnostics, Eqs. (203)-(211):

    src/topologies/refined/diagnostics.py

Matched structural ensemble runner:

    src/experiments/refined/structural.py

D041 1000-graph-per-topology structural validation: COMPLETED SUCCESSFULLY.

---

## 6. Run-level market outcomes

Implementation:

    src/experiments/refined/market_metrics.py

Evaluation sample, Eqs. (231)-(235):

    T_B = {B+1,...,T}
    W_t^(L) = {t-L+1,...,t}

Report baseline:

    B = 0

Mapping:

    Eq. (236) return_volatility(...)
    Eq. (237) rms_mispricing(...)
    Eq. (237) maximum_absolute_mispricing(...)
    Eq. (238) mean_absolute_order_flow_per_agent(...)
    Eq. (288) mean_absolute_return(...)
    Eq. (289) time_averaged_belief_variance(...)

Status: VERIFIED.

---

## 7. Rolling action covariance, Eqs. (239)-(240)

Implementation:

    src/experiments/refined/action_covariance.py

    RollingActionCovariancePoint
    rolling_action_covariance(...)

Eq. (239): average pairwise sample covariance of executed actions.

Eq. (240): exact rolling sample decomposition

    Var_hat(F)
      = sum_i Var_hat(a_i)
      + N(N-1) C_a,t

Stored `F_t=sum_i a_i,t` is validated before decomposition.

Status: VERIFIED.

---

## 8. CID components and normalisation, Eqs. (241)-(246)

Implementation:

    src/experiments/refined/cid.py

    RollingCIDComponentsPoint
    CIDReferenceScales
    CIDWeights
    RollingCIDPoint
    rolling_cid_components(...)
    standardise_cid_components(...)
    rolling_cid(...)

Mapping:

    Eq. (241) rolling sample return volatility
    Eq. (242) rolling mean of population cross-sectional belief variance
    Eq. (243) RMS signed net-order-flow pressure sqrt(mean[(F/N)^2])
    Eq. (244) explicit positive reference scales
    Eq. (245) non-negative weights summing to one
    Eq. (246) dimensionless weighted CID

Status: VERIFIED.

---

## 9. Threshold exceedance and operational stabilisation, Eqs. (247)-(250)

Implementation:

    src/experiments/refined/cid_events.py

    CIDThresholdConfiguration
    OperationalStabilisationResult
    CIDRunClassification
    classify_cid_path(...)
    operational_stabilisation(...)
    threshold_exceedance_rate(...)

Binding semantics:

    exceedance: strict >
    stabilisation admissibility: <=
    inactive guardrail: +infinity
    no qualifying stabilisation block: right-censored, no artificial time
    first-stage L_stab = 50

Status: VERIFIED.

---

## 10. Realised influence and common exposure, Eqs. (251)-(265)

Implementation:

    src/experiments/refined/influence_metrics.py

API:

    structural_hub_nodes(...)
    attention_entropy(...)
    normalised_attention_entropy(...)
    effective_number_of_sources(...)
    realised_influence_shares(...)
    realised_influence_hhi(...)
    realised_hub_influence_share(...)
    attention_overlap(...)
    attention_mobility(...)
    RealisedInfluencePoint
    RealisedInfluencePath
    realised_influence_path(...)

Mapping:

    Eqs. (251)-(253) normalised row entropy / effective source count / network means
    Eqs. (254)-(256) source influence shares and HHI
    Eq. (257) realised influence of structural hubs H_q(G)
    Eqs. (258)-(264) average pairwise attention overlap and matrix identity
    Eq. (265) RMS row-level attention mobility

Structural hubs are selected from directed in-degree in G, never from realised W_t. Cutoff ties are resolved deterministically by decreasing in-degree then increasing node label as a computational convention only.

For neutral fixed-out-degree `W_0=G/K`, tests enforce:

    s^I_j,0 = d_j^in/(NK)
    S^I_q,0 = S^G_q

Eqs. (266)-(267) KL deviation to a transition prior remain deferred with the attention-inertia extension (`tau>0`).

Status: VERIFIED at 409-test checkpoint.

---

## 11. Pre-topology market calibration protocol

Implementation:

    src/experiments/refined/market_calibration.py

Decision:

    D042 — First Refined Market-Evaluation Calibration Protocol

API:

    MarketEvaluationCalibrationProtocol
    MarketEvaluationCalibration
    first_market_evaluation_calibration_protocol(...)
    estimate_reference_scales(...)
    estimate_cid_threshold(...)
    calibrate_market_evaluation(...)

The protocol is deliberately topology-blind and two-sample:

### Scale sample

    alpha = 0
    500 replications
    seed namespace = 2026090201

Estimate:

    c_ret = pooled median(V_ret)
    c_bel = pooled median(B_bel)
    c_F   = pooled median(Q_F)

A non-positive median is a calibration failure; no epsilon patch is applied.

### Threshold sample

    alpha = 0
    500 separate replications
    seed namespace = 2026090202

Using the already-fixed reference scales and equal CID weights:

    c_CID = empirical 95th percentile of run-level peak CID

with NumPy's deterministic conservative `higher` quantile convention.

Baseline design:

    T = 1000
    B = 0
    L = 50
    robustness L = {25, 100}
    weights = (1/3,1/3,1/3)
    component guardrails = inactive
    L_stab = 50

This module freezes the calibration METHOD only. It does not yet supply numerical `c_ret`, `c_bel`, `c_F`, or `c_CID`.

Status: IMPLEMENTED; awaiting Iridis verification. Expected checkpoint: 445 tests.

---

## 12. Required gate before actual calibration simulation

Actual no-social calibration cannot be run until two additional design objects are frozen:

1. homogeneous baseline `RefinedParameters`;
2. common non-network initial-condition rule for `theta_0, b_0, x_0, p_0, R_0`.

The report permits neutral `x_0=0`, `R_0=0`, and stationary `theta_0`, but does not uniquely fix `b_0` or `p_0`. Do not invent them silently.

Legacy numerical defaults may be used only as provenance. They cannot automatically override the refined model because the refined price, trading, inventory, profit, and attention equations differ from the legacy pilot.

After those two design objects are frozen:

    build no-social calibration runner
    run small calibration smoke test
    run full 500 scale + 500 threshold samples
    persist calibration output
    freeze numerical c_ret, c_bel, c_F, c_CID
    build paired R/SW/SF market-run persistence layer
    run small paired smoke experiment
    only then submit confirmatory large Monte Carlo

---

## 13. Mechanism chain

Target measured chain:

    feasible topology G
        -> realised W_t
        -> concentration / structural-hub influence / overlap / mobility
        -> action covariance
        -> aggregate signed order flow
        -> price response / mispricing

A topology difference in volatility alone is insufficient for a strong mechanism claim.

Formal stability remains separate: equilibrium X*, complete Jacobian J*, spectral radius of J*, and Lyapunov analysis. Never use the spectral radius of row-stochastic W as the market-stability criterion.
