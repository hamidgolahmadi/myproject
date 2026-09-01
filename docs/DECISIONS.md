# Project Decisions

Last updated: 2026-09-01

This file records methodological and implementation decisions that should
remain stable across coding sessions and ChatGPT conversations.

A decision should be changed only when:

1. the doctoral report is revised;
2. a mathematical inconsistency is identified;
3. a deliberate new research design is adopted.

Any such change should be documented here and committed separately.

---

## D001 — Scientific Source of Truth

STATUS: FROZEN

The doctoral report is the scientific source of truth.

Legacy code is reference only.

If legacy code conflicts with the report:

    the report wins.

---

## D002 — Legacy Results Are Pilot Results

STATUS: FROZEN

Historical equivalence tests establish only that refactoring preserved old
behaviour.

They do NOT establish that the old model is equivalent to the refined
doctoral model.

Pilot output must not be labelled as refined-model output unless it has
been rerun under the refined equations.

---

## D003 — Separate Refined Implementation

STATUS: FROZEN

The refined model will be implemented separately under:

    src/model/refined/

Do not progressively mutate:

    baseline_env.py
    adaptive_env.py
    market_core.py

into the refined model.

Those files remain part of the legacy/pilot reproducibility layer.

---

## D004 — G and W Are Different Objects

STATUS: FROZEN

G is the binary feasible-information network.

    g_ij in {0,1}

W_t is the row-stochastic effective-attention matrix.

    w_ij,t >= 0
    sum_j w_ij,t = 1
    g_ij = 0 => w_ij,t = 0

Therefore:

    G != W

A topology generator should generate G.

Attention logic should generate W conditional on G.

---

## D005 — First-Stage G Is Exogenous and Fixed

STATUS: FROZEN

During the initial refined fixed-topology experiments:

    G is drawn before the simulation
    and remains fixed during that simulation.

Adaptive influence means:

    W_t changes

while:

    G remains fixed.

Endogenous G_t formation / rewiring is a later research stage.

---

## D006 — Equation (39) Defines Timing

STATUS: FROZEN

Canonical period order:

    W_{t-1}
        ->
    theta_t, v_t, s_t
        ->
    b_t
        ->
    vhat_t
        ->
    m_t
        ->
    desired action
        ->
    executed action
        ->
    x_t
        ->
    F_t
        ->
    p_t
        ->
    r_t
        ->
    pi_t
        ->
    R_t
        ->
    z_t
        ->
    W_t

Do not introduce within-period simultaneity unless the model itself is
formally changed.

---

## D007 — Beliefs Use Lagged Attention

STATUS: FROZEN

Beliefs satisfy:

    b_t
      =
    (1-alpha) s_t
      +
    alpha W_{t-1} b_{t-1}
      +
    epsilon_b_t

The following is prohibited in the refined baseline:

    using W_t to construct b_t.

Current reputation affects future attention and therefore future beliefs.

---

## D008 — No Within-Period Matrix Inverse

STATUS: FROZEN

The ABM does NOT solve:

    b_t = (1-alpha)s_t + alpha W_t b_t

and does not use:

    (I - alpha W_t)^(-1)

as its period-by-period belief solution.

Such matrix expressions may be used as analytical comparison devices where
the report explicitly derives them, but not as a replacement for the
sequential ABM transition.

---

## D009 — State, Belief, Value, Action, and Position Are Distinct

STATUS: FROZEN

Do not identify:

    signal = belief
    belief = perceived value
    perceived value = action
    action = position
    feasible link = effective influence weight

The canonical chain is:

    signal
        ->
    belief
        ->
    perceived value
        ->
    valuation gap
        ->
    desired trade
        ->
    executed trade
        ->
    position

---

## D010 — Desired and Executed Trades Are Different

STATUS: FROZEN

Desired trade:

    a_tilde_i,t = tanh(kappa_i m_i,t)

Executed trade is the projection of desired trade onto the
inventory-feasible interval.

Inventory feasibility must therefore be applied after the tanh response.

---

## D011 — Inventory Bound Is Separate from Action Saturation

STATUS: FROZEN

The tanh bound limits the size of one desired adjustment.

The inventory bound limits accumulated position exposure.

Both restrictions must exist.

Do not treat one as a substitute for the other.

---

## D012 — Position Dynamics

STATUS: FROZEN

Position evolves as:

    x_i,t = x_i,t-1 + a_i,t

Action is a signed flow.

Position is a stock.

---

## D013 — Net Order Flow Is Signed

STATUS: FROZEN

Aggregate order flow is:

    F_t = sum_i a_i,t

Purchases and sales can cancel.

F_t is NOT gross trading volume.

Gross volume, if required, is a separate metric:

    sum_i |a_i,t|

---

## D014 — Refined Price Equation Includes Fundamental Anchor

STATUS: FROZEN

The canonical price equation is:

    p_t - p_{t-1}
      =
    chi (v_t - p_{t-1})
      +
    lambda F_t
      +
    sigma_p epsilon_p_t

Do not use the old pure order-flow price equation as the refined baseline.

---

## D015 — Return Convention

STATUS: FROZEN

The model return is:

    r_t = p_t - p_{t-1}

where prices are interpreted in log-price or consistently normalised value
units.

Do not silently mix this with:

    (P_t - P_{t-1}) / P_{t-1}

Any future alternative convention requires corresponding changes to profit,
calibration, and interpretation.

---

## D016 — Profit Uses the Inherited Position

STATUS: FROZEN

Realised profit is:

    pi_i,t = x_i,t-1 * r_t

NOT:

    pi_i,t = x_i,t * r_t

The current action must not earn the same price movement that it helps
create.

---

## D017 — Reputation Update

STATUS: FROZEN

Reputation evolves as:

    R_i,t
      =
    gamma_R R_i,t-1
      +
    (1-gamma_R) pi_i,t

Profit-based reputation is a performance state.

It should not automatically be interpreted as pure informational skill.

---

## D018 — Initial Attention

STATUS: FROZEN

The neutral initial attention matrix is uniform over feasible neighbours:

    w_ij,0 = 1/d_i

for:

    j in N_i

and zero otherwise.

Initial W therefore depends on G.

This is correct and does not violate paired-experiment logic.

---

## D019 — Fixed-Influence Benchmark

STATUS: FROZEN

The fixed-influence benchmark uses uniform graph-supported W.

This preserves transmission through G while removing reputation-sensitive
reallocation of influence.

---

## D020 — Adaptive Attention: Frictionless First

STATUS: FROZEN FOR FIRST IMPLEMENTATION

The first adaptive implementation uses the reputation softmax:

    Equation (60)

with:

    tau_i = 0

Attention inertia from Equations (61)-(62) is a later extension.

Do not silently introduce tau > 0 into first-stage refined results.

---

## D021 — beta Does Not Mean Instability by Construction

STATUS: FROZEN

Higher beta means stronger selectivity with respect to relative reputation.

It does not mechanically imply:

    greater volatility
    greater instability
    greater topology effects

Those are simulation outcomes to be measured.

---

## D022 — alpha = 0 Is a Mandatory Negative Control

STATUS: FROZEN

At:

    alpha = 0

beliefs reduce to:

    b_t = s_t + epsilon_b_t

The network-propagation channel disappears.

Common exposure through the shared fundamental remains.

Therefore alpha=0 is a central test distinguishing:

    common exposure

from:

    social/network propagation.

---

## D023 — Correlated Order Flow Is Not Automatically Network Contagion

STATUS: FROZEN

Correlated signals, beliefs, or trades may arise from:

    common fundamental information
    common shocks
    shared exposure

without social transmission.

Mechanism claims require intermediate network/influence diagnostics.

Do not infer contagion from order-flow correlation alone.

---

## D024 — Homogeneous Benchmark Comes First

STATUS: FROZEN

The first confirmatory topology comparison uses common behavioural
parameters across agents.

Heterogeneity in:

    kappa_i
    beta_i
    tau_i
    signal precision
    other agent characteristics

is introduced only after the homogeneous structural comparison is
interpretable.

---

## D025 — Paired Confirmatory Design

STATUS: FROZEN

Use common random numbers.

Within replication r, hold common across topology classes:

    shock path
    non-network initial states
    behavioural parameters
    market parameters
    horizon
    evaluation definitions

Allow topology-specific:

    graph realization
    graph seed
    graph-supported initial W_0

The graph is the designed treatment.

---

## D026 — Random Seeds Must Have Semantic Roles

STATUS: FROZEN

Use separate seed fields such as:

    replication_id
    graph_seed
    shock_seed
    initial_state_seed
    type_assignment_seed

Do not reuse an ambiguous integer called only:

    seed

for every source of randomness.

---

## D027 — OAT Is Diagnostic, Not the Main Identification Design

STATUS: FROZEN

One-factor-at-a-time sweeps for:

    alpha
    beta
    gamma_R

remain useful diagnostics.

The main question is conditional and ultimately requires joint parameter
experiments.

---

## D028 — Mechanism Before Strong Topology Claims

STATUS: FROZEN

The central measurement chain is:

    feasible topology
        ->
    realised influence
        ->
    concentration / common-source overlap
        ->
    correlated actions
        ->
    aggregate order flow
        ->
    price response

A difference in volatility alone is insufficient to establish this
mechanism.

---

## D029 — Structural Hubs and Realised Influence Are Different

STATUS: FROZEN

A node can be structurally prominent in G without receiving a large share
of realised W_t.

Therefore report separately:

    structural hub share

and:

    realised hub influence share.

---

## D030 — CID Is Operational, Not Mathematical Stability

STATUS: FROZEN

The Composite Instability Diagnostic is a simulation diagnostic.

CID threshold crossing does NOT prove:

    divergence
    non-stationarity
    spectral radius > 1
    mathematical explosion

Use terminology:

    threshold-exceeding

and:

    operationally stabilised

unless a separate mathematical stability result justifies stronger
language.

---

## D031 — Avoid the Legacy Word "Explosive" as a Default Classification

STATUS: FROZEN FOR REFINED MODEL

Legacy output may contain:

    exploded

for historical compatibility.

New refined output should use:

    threshold_exceeded

or an equivalent explicit operational label.

"Explosive" should be reserved for a mathematically justified statement.

---

## D032 — W Spectral Radius Is Not the Stability Test

STATUS: FROZEN

A row-stochastic W naturally has spectral radius one.

This does not establish market instability.

Formal local stability concerns the complete coupled equilibrium Jacobian:

    J*

with criterion:

    spr(J*) < 1

for local asymptotic stability of the linearised discrete-time system.

---

## D033 — Simulation and Formal Stability Are Separate Tasks

STATUS: FROZEN

Simulation answers:

    what realised nonlinear paths do under finite shocks and finite horizon.

Jacobian / Lyapunov analysis answers:

    how sufficiently small perturbations behave around a specified equilibrium.

Neither should be used as a substitute for the other.

---

## D034 — Endogenous Network Formation Comes Later

STATUS: FROZEN

Do not introduce endogenous G_t before the fixed-G mechanism is validated.

The later extension will allow:

    G_t -> G_{t+1}

through economic link / rewiring decisions.

This is distinct from adaptive W_t conditional on fixed G.

---

## D035 — EKF Comes After Structural Model Validation

STATUS: FROZEN

Do not begin empirical EKF estimation before the refined state transition
and observation structure are stable.

First:

    refined model

then:

    synthetic data

then:

    parameter recovery

then:

    real-data estimation and out-of-sample validation.

---

## D036 — MARL Is Not a Current Priority

STATUS: FROZEN

Multi-agent reinforcement learning is a later optional extension.

It must not precede:

    structural identification
    fixed-topology mechanism validation
    endogenous network formation
    planner benchmark

unless the research programme is deliberately revised.

---

## D037 — No Large Monte Carlo Before Core Validation

STATUS: FROZEN

Do not submit a large refined-model Iridis experiment until:

    one-period deterministic tests pass
    multi-period deterministic tests pass
    alpha=0 negative control passes
    topology-support tests pass

Computational scale must not substitute for implementation validation.

---

## D038 — Preserve Legacy Code, Do Not Copy Legacy Bugs

STATUS: FROZEN

Legacy OAT / interaction behaviour was preserved during restructuring for
reproducibility.

Known historical quirks belong only to that legacy layer.

The refined implementation should not deliberately reproduce legacy bugs
unless required for a specifically labelled historical-comparison test.

---

## D039 — Git Discipline

STATUS: FROZEN

Make small conceptual commits.

Examples:

    Add refined state and parameter objects
    Implement refined fundamental and signal block
    Implement lagged social belief update
    Implement inventory-constrained trading
    Implement refined price and reputation transition
    Add refined one-period integration tests

Do not combine unrelated economics changes, architecture changes, and
analysis changes in one commit.

---

## D040 — Session Handoff Rule

STATUS: FROZEN

At the end of each substantial coding session:

1. update `docs/PROJECT_STATE.md`;
2. update `docs/IMPLEMENTATION_MAP.md` if mapping changed;
3. update this file if a methodological decision changed;
4. run relevant tests;
5. commit the session checkpoint.

A new conversation should be able to continue the project by reading these
three files without relying on conversational memory.

---

## D041 — First Structural-Validation Calibration

STATUS: FROZEN FOR FIRST STRUCTURAL VALIDATION RUN

Before running refined market-outcome Monte Carlo comparisons, validate the
three benchmark graph ensembles structurally using one explicit matched
configuration:

    N = 100
    K = 6
    n_graph_replications = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0
    experiment_seed = 20260901

Provenance and interpretation:

- `N = 100`, `K = 6`, and 1000 graph realisations per topology match the
  first-stage baseline scale reported in the doctoral report.
- `q = 5` means the structural hub-share diagnostic tracks the top five nodes,
  i.e. five percent of the N=100 population. This matches the report's worked
  hub-share example and the legacy structural diagnostic `top5_share`.
- The report requires a small positive Small-World rewiring probability but
  does not uniquely calibrate it numerically. `p_sw = 0.02` is retained from
  the legacy extreme structural-validation pipeline because its purpose was
  explicitly to preserve strong clustering while adding a small number of
  shortcuts. It is a calibration choice, not a report equation.
- The report requires `a0 > 0` but does not uniquely calibrate it numerically.
  `a0 = 1.0` is consistent with the report's numerical illustration and with
  the legacy linear preferential-attractiveness baseline, which starts every
  node at unit attractiveness. It is a calibration choice, not a theorem.
- `experiment_seed = 20260901` is only a reproducibility namespace seed. It
  has no economic interpretation.

Structural acceptance remains empirical at the ensemble level. In particular,
we require the realised hub-dominated ensemble to show materially greater
in-degree inequality and top-five hub share than the matched Random ensemble,
and the realised Small-World ensemble to show materially greater clustering
than the matched Random ensemble while retaining short paths. The report does
not supply universal numerical cutoffs, so do not invent a hard pass/fail
threshold before inspecting the full distributions.

If this calibration fails to generate the intended architectural treatments,
do not proceed to market-outcome interpretation. Recalibration must be
explicitly documented as a new design decision rather than silently tuned.
