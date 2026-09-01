# Project State

Last updated: 2026-09-01

## 1. Project Identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Primary development branch:

    refined-model

Stable restructuring checkpoint:

    git tag: restructuring-complete

The doctoral report `report1_25_08_2026.pdf` is the scientific source of
truth. Legacy code is reference/reproducibility code only.

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

Refined experiment-design layer:

    src/experiments/refined/
        __init__.py
        paired.py

Refined topology layer:

    src/topologies/refined/
        __init__.py
        generators.py

Legacy files under `src/model/`, `src/experiments/`, and `src/topologies/`
outside the refined namespaces must not override the report specification.

---

## 4. Frozen Scientific Decisions

The following remain binding:

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
- Paired topology comparisons reuse the same shock path and non-network
  randomness within each replication.
- Random seeds have semantic roles; graph randomness is topology-specific.
- Formal stability later concerns the complete equilibrium Jacobian `J*`, not
  the spectral radius of `W`.

See `docs/DECISIONS.md` for the full decision register.

---

## 5. Refined Core Status

The refined fixed-topology runtime corresponding to Equations (35)-(82) is
implemented and verified, including:

- parameter and state objects;
- graph/attention validation;
- fundamental and private-signal block;
- lagged social belief update;
- reputation-sensitive attention;
- perceived value, valuation gaps, bounded desired trades, inventory
  projection, positions, and signed order flow;
- refined price, return, profit, and reputation equations;
- canonical one-period transition with Equation (39) timing;
- deterministic multi-period simulation;
- explicit shock-path generation with separated RNG substreams.

The canonical simulator contains no duplicate economic equations: it repeatedly
calls `transition_one_period(...)` on explicit `PeriodShocks` objects.

---

## 6. Paired Experiment-Design Status

Implemented and VERIFIED on Iridis:

    src/experiments/refined/paired.py

with:

    ReplicationSeeds
    PairedReplicationPlan
    prepare_paired_replication(...)

Within one replication the following are common across topology treatments:

    shock_seed
    initial_state_seed
    type_assignment_seed
    realised shock_path

Each named topology receives its own deterministic `graph_seed`. Graph-seed
assignment is invariant to the ordering of topology labels in configuration.

This layer prepares random inputs only; it does not generate graphs or run a
Monte Carlo experiment.

---

## 7. Refined Topology Generators

NEWLY IMPLEMENTED under:

    src/topologies/refined/generators.py

The implementation follows Section 5.3 of the report rather than legacy
topology code.

Common benchmark restrictions from Equation (191):

    directed G
    g_ii = 0
    no duplicate directed edges
    exactly K outgoing links per agent
    total links = N K

Implemented generators:

    generate_random_fixed_out_degree(...)

For each agent, choose exactly K distinct sources uniformly without replacement
from the other N-1 agents. This is NOT an unconstrained Erdos-Renyi graph.

    generate_small_world(...)

Start from the directed ring lattice with K/2 nearest neighbours on each side
(K even), then independently rewire outgoing lattice edges with probability
`p_sw` while preserving source, no-self-link, no-duplicate, and exact row
out-degree restrictions.

    generate_hub_dominated(...)

Allocate N*K directed edge slots in random source order. For an eligible target
j, attachment probability is proportional to:

    in_degree_j + a0

with `a0 > 0`, matching Equations (198)-(201). The report's SF/Scale-Free label
is interpreted as a finite hub-dominated benchmark unless a separate power-law
diagnostic supports a stronger claim.

All public generator APIs take `graph_seed` and structural parameters only;
they do not accept `shock_seed`, `initial_state_seed`, or
`type_assignment_seed`.

The topology-generator tests are committed but AWAIT IRIDIS verification.

---

## 8. Verified Test Checkpoints

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

Latest verified checkpoint:

    138 passed in 2.06s

with a clean working tree and branch up to date with `origin/refined-model`.

The new topology-generator test file adds 28 pytest cases and has not yet been
run on Iridis.

---

## 9. Computational Milestones

Milestone 1 — deterministic one-period transition: VERIFIED

Milestone 2 — deterministic multi-period simulator: VERIFIED

Milestone 3 — multi-period alpha=0 network-null test: VERIFIED

Milestone 4 — explicit shock-path generation / CRN: VERIFIED

Milestone 5 — semantic paired replication plan: VERIFIED

Milestone 6 — refined benchmark topology generators:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

No large refined Monte Carlo experiment should be submitted yet.

---

## 10. Immediate Next Step

NEXT STEP:

1. Pull the latest `refined-model` branch on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total if the topology generators are correct:

       166 passed

4. If all 166 pass, record Milestone 6 as VERIFIED.
5. Then implement graph-supported uniform initial attention
   `W_0 = W_unif(G)` as an explicit reusable helper if not already exposed,
   and construct a refined paired topology treatment object that combines:

       paired replication plan
       topology-specific graph seed
       generated G
       graph-supported W_0
       common non-network initial conditions

6. Add an end-to-end paired-treatment construction test, including the
   `alpha=0` negative control across the generated benchmark graphs.

Do not start a large Monte Carlo run until paired treatment construction and
structural graph validation are explicit and tested.

---

## 11. Planned Development Sequence

    Phase 1  Refined fixed-topology core model, Equations (35)-(82)   COMPLETE
    Phase 2  Refined binary topology generators, G separated from W   IN PROGRESS
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

## 12. New-Chat Handoff Prompt

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
