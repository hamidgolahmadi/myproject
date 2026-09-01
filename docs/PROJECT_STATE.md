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

After the 166-test Iridis checkpoint, the hub-dominated generator was further
aligned with Equation (212): formation and node relabelling now use separate
child RNG streams derived from `graph_seed`, and the completed hub graph is
randomly relabelled so arbitrary numerical node labels are not mechanically
attached to hub positions.

This relabelling change is committed but awaits the next Iridis verification.

---

## 8. Explicit Neutral Initial Attention

NEWLY IMPLEMENTED in:

    src/model/refined/attention.py

Function:

    uniform_attention_from_graph(G)

It implements Equations (41), (225)-(226): every feasible source in row i
receives weight `1 / d_i`; unsupported entries receive zero. For benchmark
graphs with fixed out-degree K, every supported initial weight is `1 / K`.

Three direct tests were added.

---

## 9. Paired Topology Treatment Construction

NEWLY IMPLEMENTED in:

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

Each prepared treatment is simulation-ready and contains:

    topology specification
    graph seed
    generated G
    initial RefinedState
    common realised shock path
    common parameter object

The treatment builder performs NO random sampling of non-network initial
conditions and runs NO simulation.

### Important initial-condition scope

The report fixes the pairing rule for non-network initial conditions and gives
a neutral rule for some components: positions and reputation may start at zero,
`theta_0` may be drawn from its stationary distribution, and `W_0(G)` is
uniform on feasible support.

However, the report does not uniquely fix numerical rules for every component,
especially `b_0` and `p_0`. Therefore the current treatment builder does not
silently invent values such as `b_0 = theta_0 1` or `p_0 = v_0`.

Instead, `NonNetworkInitialConditions` is supplied explicitly and identically
to all paired topology treatments. The already-reserved `initial_state_seed`
remains unused until an explicit initial-condition generation rule is adopted
and documented.

---

## 10. Generated-Treatment Negative Control

A new end-to-end integration test constructs actual Random, Small-World, and
hub-dominated treatments from the paired plan and runs them with `alpha = 0`.

Required result:

- theta paths equal;
- beliefs equal;
- positions/actions equal;
- net order flow equal;
- prices/returns equal;
- profits/reputations equal;

while graph-supported attention and reputation-score matrices are allowed to
differ because the graphs differ. This is the report-defined network-null
control on the full generated-treatment pipeline.

---

## 11. Verified Test Checkpoints

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

Latest verified checkpoint:

    166 passed in 1.56s

with a clean working tree and branch up to date with `origin/refined-model`.

New changes after that checkpoint add:

    3 test cases   explicit uniform W_0(G)
    29 test cases  paired treatment construction/validation
    1 test case    generated R/SW/SF alpha=0 end-to-end control

Expected next refined total:

    199 passed

The 199-test checkpoint has NOT yet been verified on Iridis.

---

## 12. Computational Milestones

Milestone 1 — deterministic one-period transition: VERIFIED

Milestone 2 — deterministic multi-period simulator: VERIFIED

Milestone 3 — multi-period alpha=0 network-null test: VERIFIED

Milestone 4 — explicit shock-path generation / CRN: VERIFIED

Milestone 5 — semantic paired replication plan: VERIFIED

Milestone 6 — refined benchmark topology generators: VERIFIED at 166 tests;
additional Eq. (212) hub relabelling awaits re-verification

Milestone 7 — paired topology treatment construction:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

No large refined Monte Carlo experiment should be submitted yet.

---

## 13. Immediate Next Step

NEXT STEP:

1. Pull the latest `refined-model` branch on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       199 passed

4. If all 199 pass, record Milestone 7 as VERIFIED and the Equation (212)
   hub-relabel change as re-verified.
5. Then implement structural graph diagnostics required by Section 5.3.1 before
   running market-outcome Monte Carlo comparisons. At minimum begin with:

       in-degree distribution
       in-degree Gini
       structural hub share
       clustering
       component structure
       average path length on the largest connected component

6. Validate that the generated ensembles exhibit the intended structural
   separation before interpreting any topology outcome difference.

Do not start a large Monte Carlo run yet.

---

## 14. Planned Development Sequence

    Phase 1  Refined fixed-topology core model, Equations (35)-(82)   COMPLETE
    Phase 2  Refined binary topology generators, G separated from W   COMPLETE/VERIFYING
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

## 15. New-Chat Handoff Prompt

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
