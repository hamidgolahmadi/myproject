# Refined Model Implementation Map

Last updated: 2026-09-01

## 1. Source of truth

This file maps the doctoral report into the current refined codebase.

Scientific source of truth:

    report1_25_08_2026.pdf

If this map conflicts with the report, the report wins. Legacy code is reference/reproducibility only.

---

## 2. Architecture

Core economic runtime:

    src/model/refined/
        parameters.py
        state.py
        shocks.py
        fundamentals.py
        beliefs.py
        attention.py
        trading.py
        market.py
        reputation.py
        transition.py
        simulator.py

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

Rule: economic transition equations live only under `src/model/refined/`. Evaluation modules consume completed `SimulationResult` objects and must not duplicate dynamics.

---

## 3. State, shocks, and timing

Persistent state:

    X_t = (theta_t, b_t, x_t, p_t, R_t, W_t)

Implementation:

    src/model/refined/state.py
    RefinedState

Within-period outputs:

    v_t, s_t, vhat_t, m_t, desired/executed actions,
    F_t, r_t, pi_t, z_t

Implementation:

    PeriodOutputs

Innovation bundle:

    u_theta_t, epsilon_s_t, epsilon_b_t, epsilon_p_t

Implementation:

    src/model/refined/shocks.py
    PeriodShocks

Binding Eq. (39) coordinator:

    src/model/refined/transition.py
    transition_one_period(...)

Critical timing invariant:

    b_t uses W_{t-1}; W_t first affects b_{t+1}

---

## 4. Core equations (35)-(82)

### Graph and attention, Eqs. (35)-(41)

    validate_graph_support(...)
    build_neighbourhoods(...)
    validate_attention(...)
    initialise_state(...)
    uniform_attention_from_graph(...)

`G` is binary feasible support; `W_t` is row-stochastic effective attention supported by `G`.

### Fundamentals/signals, Eqs. (42)-(47)

    update_fundamental(...)
    stationary_fundamental_variance(...)
    fundamental_value(...)
    private_signals(...)
    generate_shock_path(...)

### Beliefs, Eqs. (48)-(55)

    update_beliefs(...)
    belief_noise_covariance(...)

Canonical runtime:

    b_t = (1-alpha)s_t + alpha W_{t-1} b_{t-1} + epsilon_b_t

`alpha=0` topology-null control is verified.

### Attention, Eqs. (56)-(62)

    uniform_attention_from_graph(...)
    local_reputation_statistics(...)
    standardised_reputation_scores(...)
    update_attention(...)

First-stage adaptive implementation is Eq. (60), frictionless softmax. Attention inertia/KL transition friction is deferred.

### Trading/order flow, Eqs. (63)-(73)

    perceived_values(...)
    valuation_gaps(...)
    desired_actions(...)
    inventory_feasible_bounds(...)
    execute_actions(...)
    update_positions(...)
    net_order_flow(...)

Binding:

    F_t = sum_i a_i,t

not gross volume.

Eq. (73) is evaluated through the rolling sample decomposition implemented for Eqs. (239)-(240).

### Market/reputation, Eqs. (74)-(79)

    price_change(...)
    update_price(...)
    market_return(...)
    realised_profits(...)
    update_reputation(...)

Profit uses inherited position `x_{t-1}`.

### Complete blocks, Eqs. (80)-(82)

    transition_one_period(...)

Multi-period wrapper:

    src/model/refined/simulator.py
    SimulationResult
    simulate_shock_path(...)

`states = (X_0,...,X_T)` and `period_outputs` correspond to t=1,...,T.

---

## 5. Randomness and paired design

Semantic seed architecture:

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py

Roles:

    replication_id
    graph_seed
    shock_seed
    initial_state_seed
    type_assignment_seed

Paired treatments:

    src/experiments/refined/treatments.py
    TopologySpecification
    NonNetworkInitialConditions
    PreparedTopologyTreatment
    prepare_paired_treatments(...)

Common across topology treatments within replication: shock path, non-network initial conditions, parameters, horizon, evaluation definitions.

Topology-specific: graph realization/seed and graph-supported `W_0`.

---

## 6. Benchmark topology implementation and validation

Generators:

    src/topologies/refined/generators.py
    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

Common benchmark constraints: directed binary `G`, zero diagonal, no duplicate directed edges, exactly `K` outgoing links per row, `N*K` links.

Hub-dominated formation uses `d_j^in + a0` and Eq. (212) post-formation relabelling.

Structural diagnostics, Eqs. (203)-(211):

    src/topologies/refined/diagnostics.py
    in_degree_gini(...)
    hub_link_share(...)
    symmetrised_support(...)
    global_clustering(...)
    average_path_length_lcc(...)
    largest_component_share(...)
    diagnose_graph(...)

Structural ensemble runner:

    src/experiments/refined/structural.py
    run_structural_ensemble(...)

D041 1000-graph-per-topology validation is completed and structurally separates R/SW/SF in the intended concentration/clustering dimensions.

---

## 7. Evaluation sample and run-level outcomes, Eqs. (231)-(238), (288)-(289)

Implementation:

    src/experiments/refined/market_metrics.py

Evaluation sample:

    T_B = {B+1,...,T}

Python alignment:

    period_outputs[B:T]
    states[B+1:T+1]

Baseline:

    B = 0

Mapping:

    Eq. (236) return_volatility(...)
    Eq. (237) rms_mispricing(...)
    Eq. (237) maximum_absolute_mispricing(...)
    Eq. (238) mean_absolute_order_flow_per_agent(...)
    Eq. (288) mean_absolute_return(...)
    Eq. (289) time_averaged_belief_variance(...)

Bundle:

    RunLevelMarketOutcomes
    compute_run_level_market_outcomes(...)

Status: VERIFIED at 284-test checkpoint.

---

## 8. Rolling action covariance, Eqs. (239)-(240)

Implementation:

    src/experiments/refined/action_covariance.py

Public API:

    RollingActionCovariancePoint
    rolling_action_covariance(...)

Valid endpoints:

    t = B + L, ..., T

with full rolling window:

    W_t^(L) = {t-L+1,...,t}

Eq. (239):

    C_a,t = 2/[N(N-1)] sum_{i<l} Cov_hat(a_i,a_l)

Eq. (240):

    Var_hat(F)
      = sum_i Var_hat(a_i)
      + N(N-1) C_a,t

Implementation uses sample covariance/variance with denominator `L-1`, verifies stored `F_t = sum_i a_i,t`, and checks the decomposition numerically at every endpoint.

Status: IMPLEMENTED; awaiting Iridis verification. Expected total after verification: 302 tests.

---

## 9. CID block, Eqs. (241)-(250)

Next staged implementation.

Planned mapping:

    Eqs. (241)-(243)
        rolling return volatility
        rolling belief dispersion
        RMS net-order-flow pressure

    Eqs. (244)-(246)
        fixed reference scales
        dimensionless standardisation
        weighted CID

    Eqs. (247)-(249)
        threshold-exceedance indicator
        peak CID
        exceedance-duration share

    Eq. (250)
        operational stabilisation with right-censoring

Binding design rule: `L`, reference scales, CID weights, thresholds, component guardrails, and `L_stab` must be fixed before topology-evaluation market Monte Carlo and may not be selected from observed topology rankings.

---

## 10. Realised-influence mechanisms, Eqs. (251)-(267)

Planned implementation:

    src/experiments/refined/influence_metrics.py

Quantities:

    normalised attention entropy
    effective number of sources
    realised influence column shares
    influence HHI
    realised structural-hub influence share
    attention overlap
    attention mobility

Eq. (266)-(267) KL-to-transition-prior is deferred with the attention-inertia extension because the current first-stage model is frictionless.

---

## 11. Mechanism chain

Target measurement chain:

    feasible topology G
        -> realised W_t
        -> influence concentration / overlap
        -> action covariance
        -> aggregate order flow
        -> price response / mispricing

A topology difference in volatility alone is insufficient for a strong mechanism claim.

---

## 12. Later stages

Formal stability is separate from simulation diagnostics:

    equilibrium X*
    full Jacobian J*
    spr(J*)
    Lyapunov analysis

Later: endogenous `G_t`, state-space/synthetic recovery/EKF, empirical validation, planner/policy, optional MARL.

Do not use the spectral radius of row-stochastic `W` as the market-stability criterion.

---

## 13. Current gate before large market Monte Carlo

Verified:

    core Eqs. (35)-(82)
    deterministic one/multi-period tests
    alpha=0 topology-null control
    refined topology generators
    structural diagnostics and D041 ensemble validation
    paired treatment preparation
    run-level market outcomes

Still required:

    verify rolling action covariance Eqs. (239)-(240)
    implement/verify CID Eqs. (241)-(250)
    freeze CID calibration inputs before topology evaluation
    implement/verify required realised-influence mechanism metrics

Large-scale computation must not substitute for these gates.
