# Project State

Last updated: 2026-09-01

## 1. Project Identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Primary development branch:

    refined-model

Stable restructuring checkpoint:

    git tag: restructuring-complete

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility code only.

---

## 2. Iridis Session Setup

At the beginning of an Iridis session:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model
    git status

`unset PYTHONPATH` is mandatory because the Iridis Python module injects a
system PYTHONPATH. `pytest==9.1.1` is installed in the project venv.

---

## 3. Refined Architecture

Core model:

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

Refined experiment layer:

    src/experiments/refined/
        __init__.py
        paired.py
        treatments.py

Refined topology layer:

    src/topologies/refined/
        __init__.py
        generators.py
        diagnostics.py

Legacy files outside these refined namespaces must not override the report.

---

## 4. Frozen Scientific Decisions

Binding decisions include:

- `G` is the fixed/exogenous directed binary feasible-information graph in the
  first stage.
- `W_t` is a separate row-stochastic effective-attention matrix supported by
  `G`.
- Current beliefs use inherited `W_{t-1}`, never contemporaneous `W_t`.
- No within-period `(I - alpha W_t)^(-1)` belief solution is used in the ABM.
- Desired and executed trades are distinct; inventory projection follows
  `tanh(kappa m)`.
- `F_t = sum_i a_i,t` is signed net order flow.
- Price contains both fundamental correction and order-flow impact.
- Return is `p_t - p_{t-1}` and profit uses inherited position `x_{t-1}`.
- First adaptive attention is frictionless reputation-sensitive softmax.
- `alpha = 0` is a mandatory network-propagation negative control.
- Random-number generation remains outside transition logic.
- Paired topology comparisons reuse the same realised shock path and common
  non-network initial conditions within each replication.
- Graph randomness is topology-specific and uses semantic graph seeds.
- Formal stability later concerns the complete equilibrium Jacobian `J*`, not
  the spectral radius of `W`.

See `docs/DECISIONS.md` for the full decision register.

---

## 5. Refined Core Status

The refined fixed-topology runtime corresponding to Equations (35)-(82) is
implemented and VERIFIED on Iridis, including:

- parameter/state objects;
- graph and attention validation;
- fundamental and private-signal block;
- lagged social belief update;
- reputation-sensitive attention;
- perceived value, valuation gaps, bounded desired trades, inventory
  projection, positions, and signed order flow;
- refined price, return, profit, and reputation equations;
- canonical one-period transition with Equation (39) timing;
- deterministic multi-period simulation;
- explicit shock-path generation with separated RNG substreams.

The simulator contains no duplicate economic equations; it repeatedly calls
`transition_one_period(...)` on explicit `PeriodShocks` objects.

---

## 6. Paired Replication Planning

Implemented and VERIFIED on Iridis:

    src/experiments/refined/paired.py

with:

    ReplicationSeeds
    PairedReplicationPlan
    prepare_paired_replication(...)

Common within a replication:

    shock_seed
    initial_state_seed
    type_assignment_seed
    realised shock_path

Topology-specific:

    graph_seed

Graph-seed assignment is deterministic by topology label and invariant to the
ordering of topology labels in configuration.

---

## 7. Refined Benchmark Topology Generators

Implemented and VERIFIED on Iridis under:

    src/topologies/refined/generators.py

Common Section 5.3 restrictions:

    directed G
    g_ii = 0
    no duplicate directed edges
    exactly K outgoing links per row
    total links = N K

Implemented:

    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

The Random benchmark is fixed-out-degree, not unconstrained Erdos-Renyi.
Small-World begins from the directed ring lattice and rewires outgoing edges
while preserving K. Hub-dominated uses attachment weight
`in_degree_j + a0`, `a0 > 0`.

The hub-dominated generator also implements the report's Equation (212)
post-formation random relabelling using a graph-seed-derived child stream, so
arbitrary numerical node labels are not mechanically attached to hub positions.
This correction was re-verified in the 199-test checkpoint.

---

## 8. Explicit Neutral Initial Attention

Implemented and VERIFIED on Iridis in:

    src/model/refined/attention.py

Function:

    uniform_attention_from_graph(G)

It implements Equations (41), (225)-(226): every feasible source in row i
receives weight `1 / d_i`; unsupported entries receive zero. For benchmark
graphs with fixed out-degree K, every supported initial weight is `1 / K`.

---

## 9. Paired Topology Treatment Construction

Implemented and VERIFIED on Iridis in:

    src/experiments/refined/treatments.py

Objects/functions:

    TopologySpecification
    NonNetworkInitialConditions
    PreparedTopologyTreatment
    prepare_paired_treatments(...)

The construction is:

    PairedReplicationPlan
        + topology-specific graph_seed
        + TopologySpecification
        + common NonNetworkInitialConditions
        + common RefinedParameters
            -> generated G
            -> uniform W_0(G)
            -> RefinedState
            -> PreparedTopologyTreatment

Each prepared treatment is simulation-ready and contains the topology
specification, graph seed, generated G, initial RefinedState, common realised
shock path, and common parameter object.

The treatment builder performs NO random sampling of non-network initial
conditions and runs NO simulation.

### Important initial-condition scope

The report fixes the pairing rule for non-network initial conditions and gives
a neutral rule for some components: positions and reputation may start at zero,
`theta_0` may be drawn from its stationary distribution, and `W_0(G)` is
uniform on feasible support.

The report does not uniquely fix numerical rules for every component,
especially `b_0` and `p_0`. Therefore the current treatment builder does not
silently invent values such as `b_0 = theta_0 1` or `p_0 = v_0`.

Instead, `NonNetworkInitialConditions` is supplied explicitly and identically
to all paired topology treatments. The reserved `initial_state_seed` remains
unused until an explicit initial-condition generation rule is adopted and
documented.

---

## 10. Generated-Treatment Negative Control

VERIFIED on Iridis in the 199-test checkpoint.

The end-to-end integration test constructs actual Random, Small-World, and
hub-dominated treatments from the paired plan and runs them with `alpha = 0`.

Verified equal across topology treatments:

- theta paths;
- beliefs;
- positions/actions;
- net order flow;
- prices/returns;
- profits/reputations.

Graph-supported attention and reputation-score matrices may differ because the
graphs differ, but with `alpha=0` they have no causal route into the market
outcomes.

---

## 11. Structural Graph Diagnostics

NEWLY IMPLEMENTED under:

    src/topologies/refined/diagnostics.py

The implementation follows Section 5.3.1, Equations (203)-(211), and is kept
strictly separate from market outcomes and from effective attention `W_t`.

Implemented functions:

    in_degrees(G)
    in_degree_gini(G)
    hub_link_share(G, q)
    symmetrised_support(G)
    global_clustering(G)
    largest_component_share(G)
    average_path_length_lcc(G)
    diagnose_graph(G, q=q)
    diagnose_ensemble(graphs, q=q)

and the diagnostic record:

    StructuralDiagnostics

Definitions:

- in-degree is the directed column sum of `G`;
- in-degree Gini follows Equations (203)-(205);
- hub share is the share of all incoming links received by the top-q realised
  in-degree nodes, Equation (206);
- clustering and path/component diagnostics use only the symmetrised support
  `G^u`, Equation (208), and do NOT change the directed economic model;
- global clustering is transitivity, Equation (209);
- average path length is computed only within the largest connected component,
  Equation (210);
- largest-component share is `n_max / N`, Equation (211).

The diagnostics layer deliberately does NOT hard-code the qualitative claim
that SF must always have higher concentration or SW must always have higher
clustering. The report states these are ensemble-level design expectations,
not theorems for every finite graph draw. Actual structural separation must be
checked empirically on generated ensembles before market differences are
interpreted.

Twenty direct test cases with hand-checkable graph examples are committed but
AWAIT IRIDIS verification.

---

## 12. Verified Test Checkpoints

Verified on Iridis:

    21 passed   state + parameters
    30 passed   + fundamentals + shocks
    40 passed   + beliefs
    51 passed   + trading
    66 passed   + market + reputation
    82 passed   + adaptive attention
    90 passed   + one-period transition integration
    100 passed  + deterministic multi-period simulator + alpha=0 null test
    115 passed  + shock-path generation + CRN tests
    138 passed  + semantic paired replication design
    166 passed  + refined benchmark topology generators
    199 passed  + W_0 + paired treatments + generated alpha=0 control

Latest verified checkpoint:

    199 passed in 1.87s

with a clean working tree and branch up to date with `origin/refined-model`.

New structural-diagnostic tests added after that checkpoint:

    20 test cases

Expected next refined total:

    219 passed

---

## 13. Computational Milestones

Milestone 1 — deterministic one-period transition: VERIFIED

Milestone 2 — deterministic multi-period simulator: VERIFIED

Milestone 3 — multi-period alpha=0 network-null test: VERIFIED

Milestone 4 — explicit shock-path generation / CRN: VERIFIED

Milestone 5 — semantic paired replication plan: VERIFIED

Milestone 6 — refined benchmark topology generators + Eq. (212) relabelling:
VERIFIED

Milestone 7 — paired topology treatment construction + generated-treatment
alpha=0 control: VERIFIED

Milestone 8 — report-defined structural graph diagnostics:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

No large refined market Monte Carlo experiment should be submitted yet.

---

## 14. Immediate Next Step

NEXT STEP:

1. Pull the latest `refined-model` branch on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       219 passed

4. If all 219 pass, record Milestone 8 as VERIFIED.
5. Then build a small, explicit structural-validation runner that generates
   graph ensembles only (no market simulation) and reports the distributions
   of:

       in-degree Gini
       top-q hub link share
       global clustering
       largest-component average path length
       largest-component share

   for matched Random, Small-World, and hub-dominated ensembles.
6. Use that structural-only runner to verify the intended architectural
   separation before any topology market-outcome comparison is interpreted.

Do not start a large market Monte Carlo run yet.

---

## 15. Planned Development Sequence

    Phase 1  Refined fixed-topology core model, Equations (35)-(82)   COMPLETE
    Phase 2  Refined binary topology generators, G separated from W   COMPLETE
    Phase 3  Deterministic integration and multi-period tests         COMPLETE
    Phase 4  Paired fixed-topology Monte Carlo design                 IN PROGRESS
    Phase 5  Refined market metrics and CID
    Phase 6  Influence / overlap / action-covariance diagnostics
    Phase 7  alpha / beta / gamma_R experiments and heterogeneity
    Phase 8  Endogenous feasible-network formation and rewiring
    Phase 9  Full equilibrium Jacobian, spectral radius, Lyapunov analysis
    Phase 10 State-space representation, synthetic recovery, EKF, empirical work
    Phase 11 Planner / policy analysis
    Optional later extension: MARL

---

## 16. New-Chat Handoff Prompt

When starting a new conversation:

    We are implementing my 141-page PhD report on Iridis.

    Project root:
    /iridisfs/home/hg2e25/projects/myproject

    Current branch:
    refined-model

    The doctoral report is the scientific source of truth.
    Legacy code is reference only and must not override the report.

    Read and follow:
    docs/PROJECT_STATE.md
    docs/IMPLEMENTATION_MAP.md
    docs/DECISIONS.md

    Continue from the NEXT STEP in PROJECT_STATE.md.
