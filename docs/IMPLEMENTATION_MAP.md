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

Rule: economic transition equations live only under `src/model/refined/`. Evaluation modules consume completed `SimulationResult` objects and must not duplicate dynamics.

---

## 3. Core runtime, Eqs. (35)-(82)

Persistent state:

    X_t = (theta_t, b_t, x_t, p_t, R_t, W_t)

Binding Eq. (39) coordinator:

    src/model/refined/transition.py
    transition_one_period(...)

Critical timing:

    b_t uses W_{t-1}; W_t first affects b_{t+1}

Key runtime mapping:

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

Core Eqs. (35)-(82) are VERIFIED.

---

## 4. Randomness and paired topology design

Semantic seed architecture:

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py

Roles:

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

Generated-treatment `alpha=0` topology-null control is VERIFIED.

---

## 5. Benchmark graph generation and structural validation

Generators:

    src/topologies/refined/generators.py
    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

Common benchmark constraints: directed binary `G`, zero diagonal, no duplicate directed edges, exactly `K` outgoing links per row, total links `N*K`.

Structural diagnostics, Eqs. (203)-(211):

    src/topologies/refined/diagnostics.py

Matched structural ensemble runner:

    src/experiments/refined/structural.py

D041 1000-graph-per-topology structural validation is completed successfully and structurally separates R/SW/SF in the intended dimensions.

---

## 6. Evaluation sample and run-level outcomes

Implementation:

    src/experiments/refined/market_metrics.py

Evaluation sample, Eqs. (231)-(235):

    T_B = {B+1,...,T}
    full rolling windows W_t^(L) = {t-L+1,...,t}

Baseline report choice:

    B = 0

Run-level mapping:

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

API:

    RollingActionCovariancePoint
    rolling_action_covariance(...)

Eq. (239): average pairwise sample covariance of executed actions.

Eq. (240): exact sample decomposition

    Var_hat(F)
      = sum_i Var_hat(a_i)
      + N(N-1) C_a,t

with denominator `L-1` and explicit validation that stored `F_t=sum_i a_i,t`.

Status: VERIFIED at 302-test checkpoint.

---

## 8. CID components and normalisation, Eqs. (241)-(246)

Implementation:

    src/experiments/refined/cid.py

API:

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

No reference scale is hard-coded or selected from topology rankings.

Status: VERIFIED at 341-test checkpoint.

---

## 9. Threshold exceedance and operational stabilisation, Eqs. (247)-(250)

Implementation:

    src/experiments/refined/cid_events.py

API:

    CIDThresholdConfiguration
    OperationalStabilisationResult
    CIDRunClassification
    classify_cid_path(...)
    operational_stabilisation(...)
    threshold_exceedance_rate(...)

Eq. (247): OR across `CID_t > c_CID` and active raw-component guardrails.

Eq. (248): topology-level mean of run-level exceedance indicators.

Eq. (249): peak CID and fraction of rolling windows satisfying `CID_t > c_CID`; component guardrail crossings do not enter this duration fraction.

Eq. (250): first start period with `L_stab` consecutive windows satisfying CID and every active guardrail. Exceedance uses strict `>` and stabilisation uses `<=`. Inactive guardrails map to `+infinity`. If no qualifying full block exists, stabilisation is right-censored with `stabilisation_period=None`.

First-stage report default:

    L_stab = 50

No numerical `c_CID` or component guardrail is hard-coded.

Status: VERIFIED at 374-test checkpoint.

---

## 10. Realised-influence and common-exposure mechanisms, Eqs. (251)-(265)

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

    Eq. (251) H_tilde_i,t = H(w_i,t) / log(d_i^out)
    Eq. (252) N_eff_i,t = exp(H(w_i,t))
    Eq. (253) network means of H_tilde and N_eff
    Eq. (254) s^I_j,t = (1/N) sum_i w_ij,t
    Eq. (255) source influence shares sum to one
    Eq. (256) HHI^I_t = sum_j (s^I_j,t)^2
    Eq. (257) S^I_q,t = sum_{j in H_q(G)} s^I_j,t
    Eq. (258) pairwise row-attention inner product
    Eq. (259) aggregate average pairwise overlap
    Eqs. (260)-(264) equivalent matrix/Frobenius overlap identity
    Eq. (265) M^W_t = ||W_t-W_{t-1}||_F / sqrt(N)

`H_q(G)` is always selected from directed in-degree in the feasible graph, never from realised attention. The report leaves cutoff ties unspecified; implementation resolves ties by decreasing in-degree then increasing node label as a deterministic reproducibility convention only.

For fixed-out-degree neutral uniform attention:

    W_0 = G/K

so the implementation/tests enforce:

    s^I_j,0 = d_j^in / (N K)

and therefore:

    S^I_q,0 = S^G_q.

The path evaluator reports periods `t=1,...,T`; concentration and overlap use `W_t`, while mobility uses `(W_{t-1},W_t)`, including the first change from `W_0` to `W_1`.

Agent-level entropy/effective-source arrays and source-level influence shares are retained alongside scalar network summaries.

Eqs. (266)-(267), KL deviation from a transition prior, remain DEFERRED with the attention-inertia extension because the current first-stage rule is frictionless (`tau=0`).

Status: IMPLEMENTED; awaiting Iridis verification. Expected checkpoint: 409 tests.

---

## 11. Mechanism chain

Target chain:

    feasible topology G
        -> realised W_t
        -> influence concentration / overlap
        -> action covariance
        -> aggregate order flow
        -> price response / mispricing

A topology difference in volatility alone is insufficient for a strong mechanism claim.

---

## 12. Current gate before large market Monte Carlo

Verified:

    core Eqs. (35)-(82)
    deterministic one/multi-period tests
    alpha=0 topology-null control
    refined topology generators
    structural diagnostics and D041 ensemble validation
    paired treatment preparation
    run-level market outcomes
    rolling action covariance Eqs. (239)-(240)
    CID Eqs. (241)-(246)
    threshold/stabilisation Eqs. (247)-(250)

Still required:

    verify realised-influence Eqs. (251)-(265)
    freeze market-evaluation calibration inputs before topology evaluation:
        L
        separate calibration sample/seeds
        CID reference scales
        CID weights
        c_CID
        optional component guardrails
        L_stab=50
    build/verify paired market-run persistence and a small smoke run

Calibration must be independent of observed topology rankings.

Formal stability remains separate and later: equilibrium, full Jacobian `J*`, spectral radius of `J*`, and Lyapunov analysis. Never use the spectral radius of row-stochastic `W` as the market-stability criterion.
