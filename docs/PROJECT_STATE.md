# Project State

Last updated: 2026-09-01

## 1. Project Identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Primary development branch:

    refined-model

Stable restructuring checkpoint:

    git tag: restructuring-complete

The restructuring work is complete. New doctoral-model development belongs
under `src/model/refined/` and `src/experiments/refined/`; legacy code remains
reference/reproducibility code only.

---

## 2. Scientific Source of Truth

The primary scientific source of truth is:

    report1_25_08_2026.pdf

The report, not the legacy implementation, defines the equations, event
timing, state variables, behavioural rules, topology interpretation,
experiments, diagnostics, stability analysis, and estimation roadmap.

In particular:

    old-result equivalence != refined-model correctness

---

## 3. Current Iridis Environment

At the beginning of an Iridis session use:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model
    git status

`unset PYTHONPATH` is mandatory because the Iridis Python module injects a
system PYTHONPATH that can contaminate the project virtual environment.

`pytest==9.1.1` is recorded in `requirements.txt` and installed in the current
`.venv` used for refined-model verification.

`nano` has segfaulted on Iridis; prefer heredocs for direct terminal edits.

---

## 4. Refined Architecture

Core model:

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

Refined experiment-design infrastructure:

    src/experiments/refined/
    |-- __init__.py
    `-- paired.py

Legacy modules such as `src/model/market_core.py`,
`src/model/baseline_env.py`, `src/model/adaptive_env.py`, and the existing
legacy experiment runners must not be mutated into the refined model.

---

## 5. Frozen Scientific Decisions

Binding decisions include:

- `G` is the fixed/exogenous directed binary feasible-information graph in the
  first stage.
- `W_t` is a separate row-stochastic effective-attention matrix supported by
  `G`.
- Current beliefs use inherited `W_{t-1}`, never contemporaneous `W_t`.
- No within-period `(I - alpha W_t)^(-1)` solution is used in the ABM.
- Signal, belief, perceived value, action, and position are distinct objects.
- Desired action is `tanh(kappa m)`; executed action is a separate inventory
  projection.
- `x_t = x_{t-1} + a_t`.
- `F_t = sum_i a_i,t` is signed net order flow, not gross volume.
- Price contains both the fundamental anchor and order-flow impact.
- Return is `r_t = p_t - p_{t-1}` in log-price/normalised-value units.
- Profit uses inherited positions: `pi_i,t = x_i,t-1 r_t`.
- First adaptive attention is frictionless reputation-sensitive softmax,
  Equation (60); attention inertia is deferred.
- `alpha = 0` is a mandatory network-propagation negative control while common
  fundamental exposure remains present.
- First confirmatory implementation is homogeneous.
- Random-number generation remains outside economic transition logic.
- Paired confirmatory experiments reuse the same shock path and non-network
  initial conditions across topology treatments.
- Random seeds have semantic roles; do not use one ambiguous generic `seed`.
- Formal stability later uses the complete equilibrium Jacobian, not the
  spectral radius of `W`.

See `docs/DECISIONS.md` for the full frozen decision register.

---

## 6. Implemented Refined Core

### Structural objects: Equations (35)-(41)

Implemented and tested:

- `RefinedParameters`;
- `RefinedState` and `PeriodOutputs`;
- binary graph validation and neighbourhood construction;
- graph-supported row-stochastic attention validation;
- initial-state and inventory validation.

### Exogenous information: Equations (42)-(46)

Implemented and tested:

- `PeriodShocks`;
- AR(1) fundamental update;
- stationary fundamental variance;
- fundamental-value mapping;
- private-signal construction.

### Beliefs: Equations (48)-(50)

Implemented and tested:

- homogeneous belief-noise covariance;
- lagged private-social update;
- `alpha=0` network-null behaviour.

### Reputation-sensitive attention: Equations (57)-(60)

Implemented and tested:

- local reputation mean and regularised dispersion;
- graph-supported standardised scores;
- stable graph-supported softmax;
- exact uniform weighting at `beta=0`.

### Trading and inventory: Equations (63)-(72)

Implemented and tested:

- perceived values and valuation gaps;
- desired `tanh` actions;
- inventory projection;
- position update;
- signed net order flow.

### Market and reputation: Equations (74)-(79)

Implemented and tested:

- fundamental-anchor + order-flow price equation;
- fixed return convention;
- inherited-position realised profit;
- reputation update.

### Canonical transition: Equations (39), (80)-(82)

Implemented and VERIFIED on Iridis in `src/model/refined/transition.py`.

`transition_one_period(...)` preserves the report timing and uses `W_{t-1}`
for current beliefs. `W_t` enters only the returned next state.

### Deterministic multi-period simulator

Implemented and VERIFIED on Iridis in `src/model/refined/simulator.py`.

`simulate_shock_path(...)` contains no duplicate economic equations and
repeatedly calls the canonical one-period transition.

### Shock-path generation

Implemented and VERIFIED on Iridis in `src/model/refined/shocks.py`.

`generate_shock_path(...)` takes a semantic `shock_seed` and uses independent
child streams for:

    fundamental innovations
    private-signal innovations
    belief-processing innovations
    price innovations

The generated path is explicit and reusable unchanged across topology
treatments in a paired replication.

---

## 7. Refined Paired-Experiment Infrastructure

NEWLY IMPLEMENTED under:

    src/experiments/refined/paired.py

Objects/functions:

    ReplicationSeeds
    PairedReplicationPlan
    prepare_paired_replication(...)

The design separates replication-common randomness from topology-specific
graph randomness.

Common within a replication:

    experiment_seed
    replication_id
    shock_seed
    initial_state_seed
    type_assignment_seed
    realised shock_path

Topology-specific:

    graph_seed for each named topology treatment

Graph seeds are derived deterministically from:

    experiment_seed
    replication_id
    semantic role = graph
    topology label

Therefore the graph seed assigned to a named topology is invariant to the
ordering of topology labels in a configuration file. Adding another topology
does not perturb the seeds or common shock path already assigned to existing
treatments.

This layer DOES NOT generate graphs and DOES NOT run simulations. It only
prepares auditable random-input plans for later paired treatments.

The paired-design tests are committed but AWAIT IRIDIS verification.

---

## 8. Verified Test Checkpoints

Refined checkpoints verified on Iridis:

    21 passed   state + parameters
    30 passed   + fundamentals + shocks
    40 passed   + beliefs
    51 passed   + trading
    66 passed   + market + reputation
    82 passed   + adaptive attention
    90 passed   + one-period transition integration
    100 passed  + deterministic multi-period simulator + alpha=0 null test
    115 passed  + shock-path generation + CRN tests

Latest verified checkpoint:

    115 passed in 1.70s

with a clean working tree and branch up to date with `origin/refined-model`.

The 23 newly added paired-design test cases have not yet been run on Iridis.

---

## 9. Computational Milestones

### Milestone 1 — deterministic one-period transition

    VERIFIED

### Milestone 2 — deterministic multi-period simulator

    VERIFIED

### Milestone 3 — multi-period alpha=0 network-null test

    VERIFIED

### Milestone 4 — explicit shock-path generation for paired CRN design

    VERIFIED

### Milestone 5 — semantic paired replication plan

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

No large refined Monte Carlo experiment should be submitted until the
remaining topology-generation and paired-treatment construction layers are
explicit and tested.

---

## 10. Immediate Next Step

NEXT STEP:

1. Pull the latest `refined-model` branch on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total if paired seed planning is correct:

       138 passed

4. If all 138 pass, record Milestone 5 as VERIFIED.
5. Then implement refined binary topology generators under a separate refined
   topology namespace, with `G` generated independently from `W`.
6. Each topology generator must accept only its topology-specific `graph_seed`
   and structural parameters; it must not consume `shock_seed` or other common
   replication randomness.
7. Add topology-support and reproducibility tests before constructing full
   paired topology treatments.

Do not start a large Monte Carlo run yet.

---

## 11. Planned Development Sequence

    Phase 1  Refined fixed-topology core model, Equations (35)-(82)   COMPLETE
    Phase 2  Refined binary topology generators, G separated from W
    Phase 3  Deterministic integration and multi-period tests         COMPLETE
    Phase 4  Paired fixed-topology Monte Carlo design
    Phase 5  Refined market metrics and CID
    Phase 6  Influence / overlap / action-covariance diagnostics
    Phase 7  alpha / beta / gamma_R experiments and heterogeneity
    Phase 8  Endogenous feasible-network formation and rewiring
    Phase 9  Full equilibrium Jacobian, spectral radius, Lyapunov analysis
    Phase 10 State-space representation, synthetic recovery, EKF, empirical work
    Phase 11 Planner / policy analysis
    Optional later extension: MARL

---

## 12. Definition of a Successful Refined Implementation

A refined implementation is accepted only when:

- equations match the report;
- timing matches Equation (39);
- `G` and `W` remain separate;
- `W_t` cannot affect `b_t` contemporaneously;
- inventory limits are respected;
- executed trade is distinct from desired trade;
- profit uses inherited position `x_{t-1}`;
- price contains the fundamental anchor;
- return conventions are internally consistent;
- `alpha=0` removes the network-propagation channel;
- random shocks and seeds are explicitly separated by purpose;
- paired topology comparisons use common random numbers correctly;
- deterministic tests pass before large simulations are submitted.

Historical equivalence with pilot code is not an acceptance criterion.

---

## 13. New-Chat Handoff Prompt

When starting a new conversation, use:

    We are implementing my 141-page PhD report on Iridis.

    Project root:
    /iridisfs/home/hg2e25/projects/myproject

    Current branch:
    refined-model

    The doctoral report is the scientific source of truth.
    Legacy code is reference only and must not override the report.

    Before proposing code changes, use:
    docs/PROJECT_STATE.md
    docs/IMPLEMENTATION_MAP.md
    docs/DECISIONS.md

    Continue from the NEXT STEP in PROJECT_STATE.md.

    Do not change the economics, timing, equations, experimental design,
    or terminology without grounding the change in the report.
