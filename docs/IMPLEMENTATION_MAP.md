# Refined Model Implementation Map

Last updated: 2026-09-01

## 1. Purpose

This document maps the doctoral-report model into code.

It is the implementation bridge between:

    report equation
        ->
    mathematical object
        ->
    Python module
        ->
    Python function
        ->
    validation test

The doctoral report remains the scientific source of truth.

If this file conflicts with the report, the report wins and this file must
be corrected before implementation continues.

---

# 2. Initial Implementation Scope

The FIRST implementation target is the refined fixed-topology market model:

    Equations (35)-(82)

This contains:

- network support;
- effective attention;
- state definitions;
- timing;
- initial conditions;
- fundamental process;
- private signals;
- belief formation;
- fixed and adaptive attention;
- valuation;
- bounded desired trading;
- inventory projection;
- positions;
- net order flow;
- price;
- return;
- profit;
- reputation;
- complete within-period transition.

Later analytical equations and later research extensions are deliberately
kept outside the first implementation milestone.

---

# 3. Canonical State Partition

## Persistent states

The period-t persistent state is:

    theta_t
    b_t
    x_t
    p_t
    R_t
    W_t

Suggested Python state object:

    RefinedState

located in:

    src/model/refined/state.py

Suggested fields:

    theta
    beliefs
    positions
    price
    reputation
    attention

---

## Within-period objects

The main within-period objects are:

    v_t
    s_t
    vhat_t
    m_t
    desired_actions_t
    actions_t
    F_t
    r_t
    profits_t
    z_t

These should not automatically be treated as persistent states.

They may be stored in a separate diagnostic object:

    PeriodOutputs

located in:

    src/model/refined/state.py

---

## Exogenous innovations

The period-t innovation bundle contains:

    u_theta_t
    epsilon_s_t
    epsilon_b_t
    epsilon_p_t

Suggested object:

    PeriodShocks

located in:

    src/model/refined/shocks.py

Random-number generation must be separated from economic transition logic.

---

# 4. Canonical Within-Period Timing

Equation (39) is binding for implementation.

The period transition is:

    inherited:
        theta_{t-1}
        b_{t-1}
        x_{t-1}
        p_{t-1}
        R_{t-1}
        W_{t-1}

    then:

        theta_t
        v_t
        s_t

        -> b_t

        -> vhat_t

        -> m_t

        -> desired action a_tilde

        -> executed action a_t

        -> x_t

        -> F_t

        -> p_t

        -> r_t

        -> profit pi_t

        -> R_t

        -> z_t

        -> W_t

The critical timing restriction is:

    b_t uses W_{t-1}

NOT:

    b_t uses W_t

W_t is generated at the end of period t and first affects beliefs in
period t+1.

---

# 5. Proposed Refined Package

    src/model/refined/
    |-- __init__.py
    |-- parameters.py
    |-- state.py
    |-- shocks.py
    |-- fundamentals.py
    |-- beliefs.py
    |-- attention.py
    |-- trading.py
    |-- market.py
    |-- reputation.py
    |-- transition.py
    `-- simulator.py

No existing pilot environment should be modified to become the refined
model.

The refined implementation is a separate package.

---

# 6. Equation-to-Code Map: Equations (35)-(41)

## Equation (35)

Object:

    G = [g_ij], g_ij in {0,1}

Meaning:

    directed feasible-information adjacency matrix

Implementation:

    src/model/refined/state.py
    validate_graph_support(...)

Primary topology generation will later live under:

    src/topologies/refined/

Required invariants:

    G is binary
    no unsupported attention weights
    every agent has at least one feasible information source

---

## Equation (36)

Objects:

    N_i = {j : g_ij = 1}
    d_i = |N_i|
    d_i >= 1

Implementation:

    src/model/refined/state.py
    build_neighbourhoods(G)

Outputs:

    neighbourhood masks / neighbour indices
    degree vector d

---

## Equation (37)

Object:

    W_t = [w_ij,t]

Meaning:

    effective social-attention matrix

Implementation:

    src/model/refined/state.py

W is NOT the same object as G.

---

## Equation (38)

Restrictions:

    w_ij,t >= 0
    sum_j w_ij,t = 1
    g_ij = 0 => w_ij,t = 0

Implementation:

    src/model/refined/state.py
    validate_attention(W, G)

Required test:

    test_attention_support_and_row_sums

---

## Equation (39)

Object:

    complete timing sequence

Implementation:

    src/model/refined/transition.py
    step(...)

This equation determines call ordering.

Required test:

    test_period_timing

Especially test that:

    W_t never enters b_t.

---

## Equation (40)

Initial state:

    theta_0
    b_0
    x_0
    p_0
    R_0
    W_0

Implementation:

    src/model/refined/state.py
    initialise_state(...)

---

## Equation (41)

Uniform graph-supported initial attention:

    w_ij,0 = 1/d_i   if j in N_i
             0       otherwise

Implementation:

    src/model/refined/attention.py
    uniform_attention_from_graph(G)

Required tests:

    row sums equal one
    unsupported elements equal zero
    all feasible neighbours receive equal weight

---

# 7. Equation-to-Code Map: Fundamentals and Signals

## Equation (42)

Fundamental state:

    theta_t = rho_theta * theta_{t-1} + u_theta_t

with:

    u_theta_t ~ N(0, sigma_theta^2)
    |rho_theta| < 1

Implementation:

    src/model/refined/fundamentals.py
    update_fundamental(...)

---

## Equation (43)

Stationary variance:

    Var(theta_t) = sigma_theta^2 / (1 - rho_theta^2)

Purpose:

    initialisation and analytical validation

Implementation:

    src/model/refined/fundamentals.py
    stationary_fundamental_variance(...)

This is not a separate period-transition equation.

---

## Equation (44)

Fundamental value:

    v_t = v_bar + psi * theta_t

Implementation:

    src/model/refined/fundamentals.py
    fundamental_value(...)

---

## Equation (45)

Private signal:

    s_i,t = theta_t + epsilon_s_i,t

Implementation:

    src/model/refined/fundamentals.py
    private_signals(...)

Shock generation:

    src/model/refined/shocks.py

---

## Equation (46)

Vector signal representation:

    s_t = theta_t * 1 + epsilon_s_t

Implementation:

    same runtime function as Equation (45)

Do not duplicate scalar and vector implementations.

---

## Equation (47)

Signal covariance decomposition.

Purpose:

    analytical diagnostic for common exposure

Initial implementation location:

    src/analysis/refined/belief_covariance.py

This equation is NOT required to advance the runtime simulation by one
period.

---

# 8. Equation-to-Code Map: Beliefs

## Equation (48)

Agent-level belief update:

    b_i,t
      =
    (1-alpha) s_i,t
      +
    alpha * sum_j w_ij,t-1 b_j,t-1
      +
    epsilon_b_i,t

Implementation:

    src/model/refined/beliefs.py
    update_beliefs(...)

---

## Equation (49)

Belief-noise vector and covariance.

Runtime noise generation:

    src/model/refined/shocks.py

Parameter definition:

    src/model/refined/parameters.py

---

## Equation (50)

Vector belief update:

    b_t
      =
    (1-alpha) s_t
      +
    alpha W_{t-1} b_{t-1}
      +
    epsilon_b_t

This is the canonical runtime implementation of Equations (48)-(50).

Implementation:

    src/model/refined/beliefs.py
    update_beliefs(...)

Do not create a simultaneous within-period solver.

Specifically, DO NOT implement:

    b_t = (I - alpha W_t)^(-1) ...

That expression is not the within-period solution concept of the ABM.

---

## Equation (51)

Conditional local derivative:

    partial b_t / partial b_{t-1}' = alpha W_{t-1}

Purpose:

    analytical derivative / later Jacobian work

Later implementation:

    src/stability/

Not required in the first simulator.

---

## Equations (52)-(54)

Conditional covariance decomposition.

Purpose:

    distinguish common exposure from network propagation

Later implementation:

    src/analysis/refined/belief_covariance.py

These equations are analytical diagnostics rather than state-transition
equations.

---

## Equation (55)

No-social counterfactual:

    alpha = 0

implies:

    b_t = s_t + epsilon_b_t

This is a mandatory validation condition.

Required test:

    test_alpha_zero_removes_network_channel

With identical non-network states and shocks, changing G or W must not
alter market outcomes through social transmission when alpha = 0.

---

# 9. Equation-to-Code Map: Attention

## Equation (56)

Fixed uniform effective influence:

    w_ij = 1/d_i  if j in N_i
           0      otherwise

Implementation:

    src/model/refined/attention.py
    uniform_attention_from_graph(...)

Use for the fixed-influence benchmark.

---

## Equation (57)

Neighbourhood mean reputation:

    R_bar_Ni,t

Implementation:

    src/model/refined/attention.py
    local_reputation_statistics(...)

---

## Equation (58)

Regularised local reputation dispersion:

    s_R,i,t

with positive floor:

    sigma_0 > 0

Implementation:

    src/model/refined/attention.py
    local_reputation_statistics(...)

The numerical floor is part of the model and must not be silently removed.

---

## Equation (59)

Relative reputation score:

    z_ij,t
      =
    (R_j,t - R_bar_Ni,t) / s_R,i,t

Implementation:

    src/model/refined/attention.py
    standardise_reputation(...)

Only feasible sources j in N_i are evaluated.

---

## Equation (60)

Frictionless reputation-sensitive attention:

    w_ij,t
      =
    exp(beta_i z_ij,t)
    /
    sum_{ell in N_i} exp(beta_i z_iell,t)

for j in N_i.

Outside N_i:

    w_ij,t = 0

Implementation:

    src/model/refined/attention.py
    adaptive_attention_frictionless(...)

Numerical implementation should use a stable softmax.

This is the FIRST adaptive-attention implementation.

---

## Equations (61)-(62)

Dynamic attention with inertia / KL transition friction.

These equations introduce:

    tau_i
    eta_i^W
    rho_i^W

Implementation status:

    DEFERRED EXTENSION

Planned function:

    src/model/refined/attention.py
    adaptive_attention_dynamic(...)

The first confirmatory implementation uses the nested frictionless case:

    tau_i = 0

which reduces to Equation (60).

Do not activate attention inertia silently in the first-stage simulations.

---

# 10. Equation-to-Code Map: Trading

## Equation (63)

Perceived value:

    vhat_i,t = v_bar + psi * b_i,t

Implementation:

    src/model/refined/trading.py
    perceived_values(...)

---

## Equation (64)

Valuation gap:

    m_i,t = vhat_i,t - p_{t-1}

Implementation:

    src/model/refined/trading.py
    valuation_gaps(...)

The inherited price p_{t-1} is used.

---

## Equation (65)

Conceptual chain:

    b -> vhat -> m -> a

No independent runtime equation is required.

Use as a transition-order test.

---

## Equation (66)

Desired trade:

    a_tilde_i,t = tanh(kappa_i * m_i,t)

Implementation:

    src/model/refined/trading.py
    desired_actions(...)

---

## Equation (67)

Local Taylor expansion of tanh.

Purpose:

    analytical interpretation

No separate runtime implementation.

---

## Equation (68)

Inventory-feasible action interval:

    A^x_i,t
      =
    [-xbar_i - x_i,t-1,
      xbar_i - x_i,t-1]

Implementation:

    src/model/refined/trading.py
    feasible_action_bounds(...)

---

## Equation (69)

Executed trade:

    a_i,t
      =
    projection onto A^x_i,t
    of tanh(kappa_i m_i,t)

Implementation:

    src/model/refined/trading.py
    execute_actions(...)

This equation is a key difference from the old pilot implementation.

---

## Equation (70)

Projection operator:

    Pi_[L,U](y) = min(U, max(L,y))

Implementation:

    src/model/refined/trading.py
    project_interval(...)

Required tests:

    interior action unchanged
    upper-bound action clipped
    lower-bound action clipped

---

## Equation (71)

Position update:

    x_i,t = x_i,t-1 + a_i,t

with:

    |x_i,t| <= xbar_i

Implementation:

    src/model/refined/trading.py
    update_positions(...)

Required invariant:

    position limits can never be violated.

---

## Equation (72)

Net order flow:

    F_t = sum_i a_i,t

Implementation:

    src/model/refined/trading.py
    net_order_flow(...)

F_t is signed order imbalance.

It is NOT gross trading volume.

---

## Equation (73)

Order-flow variance decomposition:

    Var(F_t)
      =
    sum_i Var(a_i,t)
      +
    2 sum_{i<j} Cov(a_i,t, a_j,t)

Purpose:

    mechanism diagnostic

Later implementation:

    src/metrics/refined/action_covariance.py

Not required for the period transition.

---

# 11. Equation-to-Code Map: Market

## Equation (74)

Price change:

    p_t - p_{t-1}
      =
    chi (v_t - p_{t-1})
      +
    lambda F_t
      +
    sigma_p epsilon_p_t

Implementation:

    src/model/refined/market.py
    update_price(...)

This is the canonical refined price equation.

The fundamental anchor must be present.

---

## Equation (75)

Equivalent level representation:

    p_t
      =
    (1-chi) p_{t-1}
      +
    chi v_t
      +
    lambda F_t
      +
    sigma_p epsilon_p_t

Implementation:

    same runtime function as Equation (74)

Do not maintain two independent price implementations.

---

## Equation (76)

Price deviation from fundamentals.

Purpose:

    mispricing diagnostic

Implementation:

    src/metrics/refined/market_outcomes.py

Not required as an independent state transition.

---

## Equation (77)

Model return:

    r_t = p_t - p_{t-1}

Implementation:

    src/model/refined/market.py
    model_return(...)

This project uses price change / consistently normalised log-price change.

Do not silently replace this with:

    (P_t - P_{t-1}) / P_{t-1}

---

# 12. Equation-to-Code Map: Profit and Reputation

## Equation (78)

Realised profit:

    pi_i,t = x_i,t-1 * r_t

Implementation:

    src/model/refined/reputation.py
    realised_profit(...)

CRITICAL:

Use inherited position:

    x_{t-1}

NOT:

    x_t

The current trade must not earn the price change that it contemporaneously
helps generate.

---

## Equation (79)

Reputation:

    R_i,t
      =
    gamma_R R_i,t-1
      +
    (1-gamma_R) pi_i,t

Implementation:

    src/model/refined/reputation.py
    update_reputation(...)

---

# 13. Equations (80)-(82): Complete Transition

## Equation (80)

Exogenous information block:

    theta_t -> v_t, s_t

Implemented through:

    fundamentals.py
    shocks.py

---

## Equation (81)

Within-period decision / market block:

    (s_t, W_{t-1}, b_{t-1}, p_{t-1}, x_{t-1})
        ->
    b_t
        ->
    vhat_t
        ->
    m_t
        ->
    a_t
        ->
    F_t
        ->
    p_t

Implementation coordinator:

    src/model/refined/transition.py
    step(...)

---

## Equation (82)

Adaptive feedback block:

    (p_t - p_{t-1}, x_{t-1})
        ->
    pi_t
        ->
    R_t
        ->
    z_t
        ->
    W_t
        ->
    b_{t+1}

Implementation coordinator:

    src/model/refined/transition.py
    step(...)

Required test:

    changing current-period profit may change W_t
    but must not change b_t retrospectively.

---

# 14. Parameter Object

All model parameters should be explicit.

Proposed file:

    src/model/refined/parameters.py

Initial parameter groups:

Fundamentals:

    rho_theta
    sigma_theta
    v_bar
    psi

Signals / beliefs:

    sigma_s
    sigma_b
    alpha

Trading:

    kappa
    x_bar

Market:

    chi
    lambda_price
    sigma_p

Reputation / attention:

    gamma_R
    beta
    sigma_0

Later extensions:

    tau
    heterogeneous beta_i
    heterogeneous kappa_i
    heterogeneous tau_i
    heterogeneous signal precision

Parameter validation should reject economically or mathematically invalid
values before a simulation starts.

---

# 15. Simulator Interface

Proposed interface:

    simulate(
        parameters,
        graph,
        initial_state,
        shock_path,
        horizon,
        attention_mode,
    )

The simulator must NOT generate hidden random shocks internally when a shock
path has been supplied.

This is necessary for paired experiments.

The simulator should return:

    state_history
    period_outputs
    metadata

---

# 16. Randomness Architecture

Randomness must be explicitly separated into:

    graph_seed
    shock_seed
    initial_state_seed
    type_assignment_seed

Do not use one ambiguous `seed` field for every source of randomness.

A paired replication must be capable of using:

    identical shock path
    identical non-network initial conditions
    identical parameter values

across different topology classes.

Graph draws remain topology-specific.

---

# 17. Required Tests Before Monte Carlo

The following tests are mandatory before any large Iridis run:

    test_graph_attention_separation
    test_attention_support_and_row_sums
    test_uniform_attention
    test_fundamental_update
    test_private_signal_mapping
    test_belief_update
    test_belief_uses_lagged_attention
    test_alpha_zero_removes_network_channel
    test_desired_action_bounds
    test_inventory_projection
    test_position_limits
    test_order_flow
    test_price_equation
    test_return_definition
    test_profit_uses_lagged_position
    test_reputation_update
    test_attention_updates_after_reputation
    test_full_one_period_transition

Then:

    deterministic multi-period integration test

Then:

    alpha = 0 topology-null test

Only after these pass should large-scale experiments begin.

---

# 18. Refined Topology Stage

After the core model passes its tests, create:

    src/topologies/refined/

Planned modules:

    random_fixed.py
    small_world.py
    hub_dominated.py
    validation.py
    attention_initialization.py

Critical rule:

    topology generator -> binary G

NOT:

    topology generator -> effective W

G defines feasible support.

W is constructed separately conditional on G.

The first-stage feasible graph remains fixed during each simulation.

---

# 19. Refined Experiment Stage

After topology validation:

    src/experiments/refined/

The confirmatory experiment will use paired/common-random-number
replications.

Within replication r:

COMMON across topology classes:

    behavioural parameters
    market parameters
    shock path
    non-network initial states
    horizon
    thresholds
    agent-type multiset, when heterogeneity is introduced

TOPOLOGY-SPECIFIC:

    graph realization
    graph seed
    graph-consistent initial W_0

The graph-generating process is the treatment.

---

# 20. Refined Metrics Stage

Planned modules:

    src/metrics/refined/market_outcomes.py
    src/metrics/refined/action_covariance.py
    src/metrics/refined/cid.py
    src/metrics/refined/influence.py
    src/metrics/refined/topology_diagnostics.py

Required separation:

    market outcomes
    mechanism diagnostics
    operational threshold classification
    formal mathematical stability

Do not collapse these concepts into one variable.

---

# 21. Mechanism Stage

The central empirical-computational chain to measure is:

    structural opportunity
        ->
    realised influence
        ->
    attention concentration / overlap
        ->
    correlated actions
        ->
    aggregate order flow
        ->
    price response

Planned quantities include:

    structural hub share
    realised hub influence share
    attention entropy
    effective number of sources
    influence HHI
    overlap
    influence mobility
    KL divergence
    action covariance
    Var(F_t)
    net order-flow pressure
    price mispricing

A topology difference without the intermediate mechanism diagnostics is not
sufficient for a strong mechanism claim.

---

# 22. Later Stages

Only after the refined fixed-topology mechanism is validated:

## Endogenous formation

    src/formation/

G_t becomes endogenous and may rewire.

## Formal stability

    src/stability/

Objects:

    equilibrium
    full Jacobian J*
    spectral radius
    Lyapunov analysis
    stability phase diagrams

## Estimation

    src/estimation/

Objects:

    state-space model
    observation equation
    synthetic parameter recovery
    EKF benchmark
    real-data estimation
    out-of-sample validation

## Policy

    src/policy/

Planner interventions follow structural identification.

MARL is optional and later.

---

# 23. Implementation Gate

The current implementation stage is COMPLETE only when:

    Equations (35)-(82) are implemented,
    one-period deterministic tests pass,
    multi-period integration tests pass,
    alpha=0 network-null test passes,
    and the new model is clearly separated from all legacy environments.

Until then:

    DO NOT submit a large refined-model Monte Carlo job.

