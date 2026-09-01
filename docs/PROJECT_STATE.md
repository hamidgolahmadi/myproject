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
under `src/model/refined/`; legacy code remains reference/reproducibility code
only.

---

## 2. Scientific Source of Truth

The primary scientific source of truth is:

    report1_25_08_2026.pdf

The report, not the legacy implementation, defines the equations, event
timing, state variables, behavioural rules, topology interpretation,
experiments, diagnostics, stability analysis, and estimation roadmap.

In particular:

    old-result equivalence != refined-model correctness

The refined executed-trade rule and refined price equation are not
algebraically equivalent to the earlier pilot implementation.

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

`pytest==9.1.1` is now recorded in `requirements.txt` and installed in the
current `.venv` used for refined-model verification.

`nano` has segfaulted on Iridis; prefer heredocs for direct terminal edits.

---

## 4. Refined Package Architecture

The refined implementation is isolated under:

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

Legacy modules such as `src/model/market_core.py`,
`src/model/baseline_env.py`, and `src/model/adaptive_env.py` must not be
mutated into the refined model.

---

## 5. Frozen Scientific Decisions

The following decisions remain binding:

- `G` is the fixed/exogenous directed binary feasible-information graph in the
  first stage.
- `W_t` is a separate row-stochastic effective-attention matrix supported by
  `G`.
- `g_ij = 0 => w_ij,t = 0`.
- Self-links are not silently prohibited because Equations (35)-(36) do not
  prohibit them.
- Current beliefs use inherited `W_{t-1}`, never contemporaneous `W_t`.
- No within-period `(I - alpha W_t)^(-1)` solution is used in the ABM.
- Signal, belief, perceived value, action, and position are distinct objects.
- Desired action is `tanh(kappa m)`; executed action is the separate inventory
  projection.
- `x_t = x_{t-1} + a_t`.
- `F_t = sum_i a_i,t` is signed net order flow, not gross volume.
- Price contains both the fundamental anchor and order-flow impact.
- Return is `r_t = p_t - p_{t-1}` in log-price/normalised-value units.
- Profit uses inherited positions: `pi_i,t = x_i,t-1 r_t`.
- Reputation follows the report's exponentially weighted update.
- First adaptive attention is frictionless reputation-sensitive softmax,
  Equation (60); attention inertia is deferred.
- `alpha = 0` is a mandatory network-propagation negative control while common
  fundamental exposure remains present.
- First confirmatory implementation is homogeneous.
- Random-number generation remains outside economic transition logic; realised
  shocks are supplied explicitly.
- Formal stability later uses the complete equilibrium Jacobian, not the
  spectral radius of `W`.

---

## 6. Implemented Refined Equations

### Structural objects: Equations (35)-(41)

Implemented and tested:

- `RefinedParameters` with first-stage homogeneous parameter validation;
- `RefinedState(theta, beliefs, positions, price, reputation, attention)`;
- `PeriodOutputs` for within-period diagnostics;
- binary graph validation and neighbourhood construction;
- graph-supported row-stochastic attention validation;
- initial-state and inventory validation.

### Exogenous information: Equations (42)-(46)

Implemented and tested:

- `PeriodShocks` as an already-realised innovation bundle;
- AR(1) fundamental update;
- stationary fundamental variance;
- fundamental-value mapping;
- private-signal construction with no hidden RNG or scaling.

### Beliefs: Equations (48)-(50)

Implemented and tested:

- homogeneous belief-noise covariance;
- lagged private-social update
  `b_t = (1-alpha)s_t + alpha W_{t-1}b_{t-1} + epsilon_b,t`;
- `alpha=0` network-null behaviour at the belief block.

### Reputation-sensitive attention: Equations (57)-(60)

Implemented and tested:

- local reputation mean;
- regularised local dispersion with `sigma_0 > 0`;
- graph-supported standardised scores `z_ij,t`;
- numerically stable graph-supported softmax;
- exact uniform weighting at `beta=0`.

### Trading and inventory: Equations (63)-(72)

Implemented and tested:

- perceived values;
- valuation gaps relative to inherited price;
- desired `tanh` actions;
- inventory-feasible action bounds;
- projection to executed trades;
- position update;
- signed net order flow.

### Market and reputation: Equations (74)-(79)

Implemented and tested:

- fundamental-anchor + order-flow price equation;
- price level update;
- fixed return convention;
- inherited-position realised profit;
- reputation update.

### Canonical transition: Equations (39), (80)-(82)

Implemented in `src/model/refined/transition.py`:

    theta_t, v_t, s_t
        -> b_t using W_{t-1}
        -> perceived value
        -> valuation gap
        -> desired action
        -> executed action
        -> x_t
        -> F_t
        -> p_t
        -> r_t
        -> pi_t using x_{t-1}
        -> R_t
        -> z_t
        -> W_t

`transition_one_period(...)` is pure with respect to the inherited state and
shock bundle. It supports:

- adaptive end-of-period attention; and
- `adaptive_attention=False` for the fixed-influence benchmark, which carries
  inherited valid attention forward unchanged.

Eight end-to-end transition tests have been added but have NOT yet been run on
Iridis at this checkpoint.

---

## 7. Verified Test Checkpoints

The following refined checkpoints have been run successfully on Iridis:

    21 passed  state + parameters
    30 passed  + fundamentals + shocks
    40 passed  + beliefs
    51 passed  + trading
    66 passed  + market + reputation
    82 passed  + adaptive attention

Latest verified checkpoint:

    82 passed in 1.42s

with a clean working tree before the transition implementation was pulled.

The newly added transition tests still require Iridis verification.

---

## 8. Immediate Next Step

NEXT STEP:

1. Pull the latest `refined-model` branch on Iridis.
2. Run the complete refined unit/integration set including
   `tests/test_refined_transition.py`.
3. Expected total if the transition integration is correct:

       90 passed

4. If all 90 tests pass, implement the second computational milestone:
   a small deterministic multi-period simulator in
   `src/model/refined/simulator.py` using an explicit supplied shock path.
5. Add multi-period timing/state-history tests before any Monte Carlo work.
6. Then strengthen the `alpha=0` paired network-null test across multiple
   periods.

Do not start large topology experiments or Monte Carlo runs yet.

---

## 9. Computational Milestones

### Milestone 1 — deterministic one-period transition

Current status:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

Acceptance requirements include:

- exact Equation (39) ordering;
- `W_{t-1}` affects current beliefs;
- `W_t` cannot affect current beliefs;
- inventory bounds hold;
- signed order flow is used;
- price includes the fundamental anchor;
- profit uses `x_{t-1}`;
- end-of-period `W_t` is carried only into the next state.

### Milestone 2 — deterministic multi-period simulator

Current status:

    NOT YET IMPLEMENTED

It must consume explicit period shocks, preserve state histories, and call the
canonical one-period transition rather than duplicate the equations.

### Milestone 3 — multi-period alpha=0 network-null test

Current status:

    NOT YET IMPLEMENTED

Under paired common shocks and identical non-network initial conditions,
changing topology must not change beliefs, actions, order flow, prices,
profits, or reputation when `alpha=0`, up to numerical tolerance. The next
period attention matrices may differ by graph support, but that channel must
remain causally inactive while `alpha=0`.

Only after these milestones pass should confirmatory topology experiments be
run.

---

## 10. Planned Development Sequence

    Phase 1  Refined fixed-topology core model, Equations (35)-(82)
    Phase 2  Refined binary topology generators, G separated from W
    Phase 3  Deterministic integration and multi-period tests
    Phase 4  Paired fixed-topology Monte Carlo design
    Phase 5  Refined market metrics and CID
    Phase 6  Influence / overlap / action-covariance mechanism diagnostics
    Phase 7  alpha / beta / gamma_R experiments and heterogeneity
    Phase 8  Endogenous feasible-network formation and rewiring
    Phase 9  Full equilibrium Jacobian, spectral radius, Lyapunov analysis
    Phase 10 State-space representation, synthetic recovery, EKF, empirical work
    Phase 11 Planner / policy analysis
    Optional later extension: MARL

---

## 11. Definition of a Successful Refined Implementation

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
- deterministic tests pass before large simulations are submitted.

Historical equivalence with pilot code is not an acceptance criterion.

---

## 12. New-Chat Handoff Prompt

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
