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

Legacy code is reference/reproducibility code only and must not override the
report.

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
        seeding.py
        paired.py
        treatments.py
        structural.py

Refined topology layer:

    src/topologies/refined/
        __init__.py
        generators.py
        diagnostics.py

---

## 4. Frozen Scientific Decisions

Binding decisions include:

- `G` is the fixed/exogenous directed binary feasible-information graph in the
  first stage.
- `W_t` is a separate row-stochastic effective-attention matrix supported by
  `G`.
- Current beliefs use inherited `W_{t-1}`, never contemporaneous `W_t`.
- Desired and executed trades are distinct; inventory projection follows
  `tanh(kappa m)`.
- `F_t = sum_i a_i,t` is signed net order flow.
- Price contains fundamental correction and order-flow impact.
- Return is `p_t - p_{t-1}` and profit uses inherited position `x_{t-1}`.
- First adaptive attention is frictionless reputation-sensitive softmax.
- `alpha = 0` is a mandatory network-propagation negative control.
- Random-number generation remains outside transition logic.
- Paired topology comparisons reuse common shocks and non-network initial
  conditions within replication; graph randomness is topology-specific.
- Formal stability later uses the complete equilibrium Jacobian `J*`, not the
  spectral radius of `W`.

See `docs/DECISIONS.md` for the full decision register.

---

## 5. Verified Refined Core

The fixed-topology runtime for Equations (35)-(82) is implemented and VERIFIED,
including:

- parameters/state;
- graph and attention validation;
- fundamentals/signals;
- lagged social belief update;
- adaptive attention;
- valuation/trading/inventory/order flow;
- price/return/profit/reputation;
- canonical Equation (39) one-period transition;
- deterministic multi-period simulation;
- explicit shock-path generation with separated RNG substreams.

The simulator contains no duplicate economic equations and repeatedly calls
`transition_one_period(...)`.

---

## 6. Verified Paired Experiment Infrastructure

Implemented and VERIFIED:

    ReplicationSeeds
    PairedReplicationPlan
    prepare_paired_replication(...)
    TopologySpecification
    NonNetworkInitialConditions
    PreparedTopologyTreatment
    prepare_paired_treatments(...)

Semantic seed derivation now lives in:

    src/experiments/refined/seeding.py

with:

    derive_semantic_seed(...)
    derive_graph_seed(...)

The seed namespace string is unchanged from the previously verified paired
implementation, so existing deterministic graph/shock seed assignments are
preserved.

The paired treatment layer generates topology-specific `G` and neutral
`W_0(G)` while keeping common non-network initial conditions, common shock
path, and common parameters across topology treatments.

The report does not uniquely fix numerical rules for every initial-state
component, especially `b_0` and `p_0`; therefore they remain explicit inputs
rather than silently invented values.

The generated Random/Small-World/hub-dominated `alpha=0` end-to-end negative
control is VERIFIED: all downstream economic paths coincide across topology
up to numerical tolerance.

---

## 7. Verified Topology Generators

Implemented and VERIFIED under:

    src/topologies/refined/generators.py

All three benchmark graphs are directed, simple, zero-diagonal, and contain
exactly `K` outgoing links per row and `N*K` total directed links.

Generators:

    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

The Random benchmark is fixed-out-degree, not unconstrained Erdos-Renyi.
Small-World begins from the directed ring lattice and rewires outgoing edges
while preserving `K`. Hub-dominated uses attachment probability proportional
to `in_degree_j + a0` and applies post-formation random node relabelling from
Equation (212).

---

## 8. Verified Structural Diagnostics

Implemented and VERIFIED under:

    src/topologies/refined/diagnostics.py

Functions:

    in_degrees(G)
    in_degree_gini(G)
    hub_link_share(G, q)
    symmetrised_support(G)
    global_clustering(G)
    largest_component_share(G)
    average_path_length_lcc(G)
    diagnose_graph(G, q=q)
    diagnose_ensemble(graphs, q=q)

These follow Section 5.3.1, Equations (203)-(211):

- concentration measures use directed in-degree of `G`;
- clustering/path/component measures use diagnostic-only symmetrised support
  `G^u`;
- APL is computed only inside the largest connected component;
- largest-component share is reported alongside APL.

The implementation does NOT hard-code qualitative topology rankings. The
report treats higher SF concentration and higher SW clustering as ensemble-level
design expectations that must be checked empirically.

---

## 9. Structural-Only Ensemble Runner

NEWLY IMPLEMENTED in:

    src/experiments/refined/structural.py

Objects/functions:

    StructuralEnsembleRecord
    DistributionSummary
    TopologyStructuralSummary
    StructuralEnsembleResult
    run_structural_ensemble(...)

Purpose:

- generate matched graph ensembles only;
- use the exact same `(experiment_seed, replication_id, topology_label)` graph
  seed mapping as later paired market replications;
- compute graph-level Section 5.3.1 diagnostics;
- preserve all raw graph-level records;
- provide descriptive mean/std/min/q25/median/q75/max summaries as a secondary
  view.

The structural runner deliberately does NOT accept or generate:

    RefinedParameters
    shocks
    shock_seed
    initial market state
    W_t
    n_periods
    market outcomes

This keeps structural validation causally and computationally separate from
market simulation.

The runner tests are committed but AWAIT IRIDIS verification.

---

## 10. Verified Test Checkpoints

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
    219 passed  + report-defined structural graph diagnostics

Latest verified checkpoint:

    219 passed in 1.91s

with a clean working tree and branch up to date with `origin/refined-model`.

New structural-ensemble/seeding tests after that checkpoint:

    28 test cases

Expected next refined total:

    247 passed

---

## 11. Computational Milestones

Milestone 1 — deterministic one-period transition: VERIFIED

Milestone 2 — deterministic multi-period simulator: VERIFIED

Milestone 3 — multi-period alpha=0 network-null test: VERIFIED

Milestone 4 — explicit shock-path generation / CRN: VERIFIED

Milestone 5 — semantic paired replication plan: VERIFIED

Milestone 6 — refined benchmark topology generators + Eq. (212): VERIFIED

Milestone 7 — paired treatment construction + generated alpha=0 control: VERIFIED

Milestone 8 — structural graph diagnostics, Eqs. (203)-(211): VERIFIED

Milestone 9 — structural-only matched ensemble runner:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

No large refined market Monte Carlo should be submitted yet.

---

## 12. Immediate Next Step

1. Pull latest `refined-model` on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       247 passed

4. If all 247 pass, record Milestone 9 as VERIFIED.
5. Then identify the report-consistent structural calibration for the baseline
   ensemble, especially `N`, `K`, `p_sw`, `a0`, and hub-count `q`.
6. Run the structural-only ensemble validation before any market Monte Carlo.
7. Inspect distributions of:

       in-degree Gini
       top-q hub link share
       global clustering
       average path length on LCC
       largest-component share

8. Only if the intended architecture-level separation is demonstrated should
   market-outcome topology comparisons proceed.

The report records baseline `N = 100`, average out-degree `K = 6`, and 1,000
network realisations in the retained first-stage experiment. The exact refined
`p_sw`, `a0`, and `q` calibration must be grounded explicitly before the new
confirmatory structural run rather than guessed.

---

## 13. Planned Development Sequence

    Phase 1  Refined fixed-topology core model                         COMPLETE
    Phase 2  Refined binary topology generators                       COMPLETE
    Phase 3  Deterministic integration and multi-period tests         COMPLETE
    Phase 4  Paired fixed-topology Monte Carlo design                 IN PROGRESS
    Phase 5  Refined market metrics and CID
    Phase 6  Influence / overlap / action-covariance diagnostics
    Phase 7  alpha / beta / gamma_R experiments and heterogeneity
    Phase 8  Endogenous feasible-network formation and rewiring
    Phase 9  Full equilibrium Jacobian / Lyapunov analysis
    Phase 10 State-space / synthetic recovery / EKF / empirical work
    Phase 11 Planner / policy analysis

Optional later extension: MARL

---

## 14. New-Chat Handoff Prompt

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
