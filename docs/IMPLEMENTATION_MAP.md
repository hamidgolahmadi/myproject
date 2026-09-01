# Refined Model Implementation Map

Last updated: 2026-09-01

## 1. Purpose and source of truth

This document maps the doctoral report into the current refined codebase:

    report equation / design object
        -> Python module
        -> public function / object
        -> validation layer

Scientific source of truth:

    report1_25_08_2026.pdf

If this map conflicts with the report, the report wins.

Legacy code is reproducibility/reference code only.

---

## 2. Canonical refined architecture

Core economic model:

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

Topology support and structural validation:

    src/topologies/refined/
        generators.py
        diagnostics.py

Paired experiment / evaluation layer:

    src/experiments/refined/
        seeding.py
        paired.py
        treatments.py
        structural.py
        calibration.py
        structural_io.py
        market_metrics.py

The economic transition is implemented only in `src/model/refined/`. Evaluation modules consume completed simulation output and must not duplicate economic dynamics.

---

## 3. Canonical state and timing

Persistent state at date t:

    X_t = (theta_t, b_t, x_t, p_t, R_t, W_t)

Implementation:

    src/model/refined/state.py
    RefinedState

Within-period non-persistent outputs:

    v_t
    s_t
    vhat_t
    m_t
    desired action
    executed action
    F_t
    r_t
    pi_t
    z_t

Implementation:

    src/model/refined/state.py
    PeriodOutputs

Exogenous innovation bundle:

    u_theta_t
    epsilon_s_t
    epsilon_b_t
    epsilon_p_t

Implementation:

    src/model/refined/shocks.py
    PeriodShocks

Binding Equation (39) order:

    inherited W_{t-1}, theta_{t-1}, b_{t-1}, x_{t-1}, p_{t-1}, R_{t-1}
        -> theta_t, v_t, s_t
        -> b_t
        -> vhat_t
        -> m_t
        -> desired action
        -> executed action
        -> x_t
        -> F_t
        -> p_t
        -> r_t
        -> pi_t
        -> R_t
        -> z_t
        -> W_t

Implementation coordinator:

    src/model/refined/transition.py
    transition_one_period(...)

Critical invariant:

    b_t uses W_{t-1}
    W_t first affects b_{t+1}

---

## 4. Equations (35)-(41): graph, attention support, initial state

### Eq. (35): binary directed feasible graph G

Implementation:

    src/model/refined/state.py
    validate_graph_support(...)

Benchmark generation:

    src/topologies/refined/generators.py

### Eq. (36): neighbourhoods and row degree

Implementation:

    src/model/refined/state.py
    build_neighbourhoods(...)

### Eqs. (37)-(38): effective attention W and graph support

Implementation:

    src/model/refined/state.py
    validate_attention(...)

Binding distinction:

    G != W

### Eq. (39): timing

Implementation:

    src/model/refined/transition.py
    transition_one_period(...)

### Eq. (40): initial state

Implementation:

    src/model/refined/state.py
    initialise_state(...)

### Eq. (41): neutral graph-supported W_0

Implementation:

    src/model/refined/attention.py
    uniform_attention_from_graph(...)

---

## 5. Equations (42)-(47): fundamentals and private signals

### Eq. (42): AR(1) fundamental state

    src/model/refined/fundamentals.py
    update_fundamental(...)

### Eq. (43): stationary fundamental variance

    src/model/refined/fundamentals.py
    stationary_fundamental_variance(...)

### Eq. (44): fundamental value v_t

    src/model/refined/fundamentals.py
    fundamental_value(...)

### Eqs. (45)-(46): private signals

    src/model/refined/fundamentals.py
    private_signals(...)

Shock-path generation:

    src/model/refined/shocks.py
    generate_shock_path(...)

### Eq. (47): common-fundamental signal covariance

Analytical interpretation only at current stage; no duplicate runtime equation.

---

## 6. Equations (48)-(55): beliefs

### Eqs. (48)-(50): lagged social belief update

    src/model/refined/beliefs.py
    update_beliefs(...)

Canonical form:

    b_t = (1-alpha)s_t + alpha W_{t-1} b_{t-1} + epsilon_b_t

### Eq. (49): belief-noise covariance

    src/model/refined/beliefs.py
    belief_noise_covariance(...)

### Eq. (51): local derivative alpha W_{t-1}

Reserved for later Jacobian work; not a separate runtime transition.

### Eqs. (52)-(54): covariance decomposition

Analytical mechanism diagnostics; deferred.

### Eq. (55): alpha=0 exclusion control

Verified in deterministic and generated-treatment multi-period tests.

---

## 7. Equations (56)-(62): effective attention

### Eq. (56): fixed uniform graph-supported W

    src/model/refined/attention.py
    uniform_attention_from_graph(...)

### Eqs. (57)-(58): local reputation mean and dispersion

    src/model/refined/attention.py
    local_reputation_statistics(...)

### Eq. (59): relative reputation scores z_ij,t

    src/model/refined/attention.py
    standardised_reputation_scores(...)

### Eq. (60): frictionless reputation softmax

    src/model/refined/attention.py
    update_attention(...)

Current first-stage implementation:

    tau = 0

### Eqs. (61)-(62): transition/inertia extension

Deferred. Must not be activated silently in first-stage results.

---

## 8. Equations (63)-(73): valuation, trading, inventory, order flow

### Eq. (63): perceived value

    src/model/refined/trading.py
    perceived_values(...)

### Eq. (64): valuation gap using p_{t-1}

    src/model/refined/trading.py
    valuation_gaps(...)

### Eq. (66): desired tanh action

    src/model/refined/trading.py
    desired_actions(...)

### Eq. (68): inventory-feasible interval

    src/model/refined/trading.py
    inventory_feasible_bounds(...)

### Eqs. (69)-(70): projected executed action

    src/model/refined/trading.py
    execute_actions(...)

### Eq. (71): position update

    src/model/refined/trading.py
    update_positions(...)

### Eq. (72): signed net order flow

    src/model/refined/trading.py
    net_order_flow(...)

Binding interpretation:

    F_t = sum_i a_i,t

not gross volume.

### Eq. (73): action covariance / order-flow variance decomposition

Next evaluation implementation stage; see Eqs. (239)-(240) below.

---

## 9. Equations (74)-(79): market, return, profit, reputation

### Eqs. (74)-(75): anchored price law

    src/model/refined/market.py
    price_change(...)
    update_price(...)

### Eq. (76): price deviation / mispricing

Not a separate transition. Evaluated from contemporaneous `state.price - period_output.fundamental_value`.

### Eq. (77): model return

    src/model/refined/market.py
    market_return(...)

### Eq. (78): profit using inherited position

    src/model/refined/reputation.py
    realised_profits(...)

Binding:

    pi_i,t = x_i,t-1 * r_t

### Eq. (79): reputation update

    src/model/refined/reputation.py
    update_reputation(...)

---

## 10. Equations (80)-(82): complete transition blocks

### Eq. (80): exogenous information block

    src/model/refined/fundamentals.py
    src/model/refined/shocks.py

### Eq. (81): within-period decision/market block

    src/model/refined/transition.py
    transition_one_period(...)

### Eq. (82): adaptive feedback block

    src/model/refined/transition.py
    transition_one_period(...)

Current-period reputation updates W_t; W_t cannot retrospectively alter b_t.

---

## 11. Multi-period simulation and randomness

Canonical simulator:

    src/model/refined/simulator.py
    SimulationResult
    simulate_shock_path(...)

`SimulationResult.states` contains:

    X_0, X_1, ..., X_T

and `period_outputs` contains the T transition outputs for t=1,...,T.

Randomness architecture:

    src/experiments/refined/seeding.py
    src/experiments/refined/paired.py

Semantic roles:

    replication_id
    graph_seed
    shock_seed
    initial_state_seed
    type_assignment_seed

Common random numbers are reused across topology treatments inside a paired replication.

---

## 12. Section 5.3 benchmark graph implementation

Report constraints around Eq. (191):

    directed binary G
    zero diagonal for benchmark generators
    no duplicate directed edges
    exactly K outgoing links per row
    total links = N*K

Implementation:

    src/topologies/refined/generators.py

Functions:

    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

Hub-dominated formation uses `d_j^in + a0` and applies Eq. (212) random relabelling after formation.

---

## 13. Equations (203)-(211): structural graph diagnostics

Implementation:

    src/topologies/refined/diagnostics.py

Mapping:

    Eqs. (203)-(205)  in_degree_gini(...)
    Eq. (206)         hub_link_share(...)
    Eq. (208)         symmetrised_support(...)
    Eq. (209)         global_clustering(...)
    Eq. (210)         average_path_length_lcc(...)
    Eq. (211)         largest_component_share(...)

Bundle:

    StructuralDiagnostics
    diagnose_graph(...)

Matched ensemble runner:

    src/experiments/refined/structural.py
    run_structural_ensemble(...)

D041 structural validation with 1000 graphs/topology has been completed successfully.

---

## 14. Paired topology-treatment construction

Implementation:

    src/experiments/refined/treatments.py

Objects:

    TopologySpecification
    NonNetworkInitialConditions
    PreparedTopologyTreatment
    prepare_paired_treatments(...)

Construction:

    paired replication plan
        + topology-specific graph_seed
        + topology specification
        + common non-network initial conditions
        -> G
        -> uniform W_0(G)
        -> RefinedState
        -> simulation-ready treatment

No simulation is run during treatment construction.

---

## 15. Equations (231)-(238): evaluation sample and principal run outcomes

Current implementation:

    src/experiments/refined/market_metrics.py

Evaluation sample:

    T_B = {B+1,...,T}

Python alignment:

    period_outputs[B:T]
    states[B+1:T+1]

Baseline report choice:

    B = 0

### Eq. (236): return volatility RV

    return_volatility(...)

Uses sample standard deviation with denominator `|T_B|-1`.

### Eq. (237): RMS mispricing RMSM

    rms_mispricing(...)

Uses contemporaneous `p_t - v_t`.

### Eq. (237): maximum absolute mispricing MAM

    maximum_absolute_mispricing(...)

### Eq. (238): mean absolute order flow per agent MAF

    mean_absolute_order_flow_per_agent(...)

Uses:

    mean_t |F_t| / N

not gross action volume.

Bundle:

    RunLevelMarketOutcomes
    compute_run_level_market_outcomes(...)

Status:

    IMPLEMENTED; awaiting Iridis verification.

---

## 16. Equations (288)-(289): retained sensitivity outcomes

Implementation in the same evaluation module:

    Eq. (288) mean_absolute_return(...)
    Eq. (289) time_averaged_belief_variance(...)

Belief variance uses the population cross-sectional denominator N at each t and then averages across the evaluation sample.

Status:

    IMPLEMENTED; awaiting Iridis verification.

---

## 17. Equations (239)-(250): action covariance and CID

Next staged implementation.

Planned mapping:

    Eqs. (239)-(240)
        rolling average pairwise action covariance
        exact rolling Var(F) decomposition

    Eqs. (241)-(243)
        rolling return volatility
        rolling belief dispersion
        RMS net-order-flow pressure

    Eqs. (244)-(246)
        fixed reference-scale standardisation
        dimensionless CID

    Eqs. (247)-(249)
        threshold-exceedance indicator
        peak CID
        exceedance duration share

    Eq. (250)
        operational stabilisation with right-censoring

Important design rule:

    reference scales, thresholds, guardrails, L, and L_stab
    must be fixed before topology evaluation

and may not be chosen after observing topology rankings.

---

## 18. Equations (251)-(267): realised influence mechanisms

Planned after/alongside CID implementation:

    normalised attention entropy
    effective number of sources
    network-average entropy / effective sources
    realised influence column shares
    influence HHI
    realised hub influence share
    attention overlap
    attention mobility
    KL deviation from transition prior when the dynamic-attention extension is active

Likely implementation location:

    src/experiments/refined/influence_metrics.py

The current frictionless first-stage model does not yet use the Eq. (266)-(267) transition-prior KL term because attention inertia is deferred.

---

## 19. Mechanism chain

The report's computational mechanism chain remains:

    feasible topology G
        -> realised W_t
        -> influence concentration / overlap
        -> action covariance
        -> aggregate order flow
        -> price response / mispricing

A topology difference in volatility alone is insufficient for a strong mechanism claim.

---

## 20. Formal stability and later extensions

Formal local stability is separate from simulation diagnostics.

Later location:

    src/stability/

Required objects:

    equilibrium X*
    complete Jacobian J*
    spectral radius spr(J*)
    Lyapunov equation / local stability analysis

Do not use the spectral radius of row-stochastic W as the market stability test.

Endogenous G_t formation, EKF/state-space estimation, planner analysis, and optional MARL remain later stages.

---

## 21. Current implementation gate

Verified before market Monte Carlo:

    core Eqs. (35)-(82)
    deterministic one-period transition
    deterministic multi-period transition
    alpha=0 topology-null control
    refined topology generators
    structural graph diagnostics
    paired treatment preparation
    D041 structural ensemble separation

Still required before large refined market Monte Carlo:

    verify run-level market outcome metrics
    implement/verify rolling action covariance
    implement/verify CID definitions
    fix CID calibration inputs before topology evaluation
    implement/verify required realised-influence mechanism metrics

Large-scale computation must not substitute for these implementation gates.
