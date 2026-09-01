# Project State

Last updated: 2026-09-01

## 1. Project Identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Primary development branch:

    refined-model

Stable restructuring checkpoint:

    git tag: restructuring-complete

The restructuring work has been merged into `main`.
All implementation of the refined doctoral model should now be developed on
`refined-model` or on descendant feature branches created from it.

---

## 2. Scientific Source of Truth

The primary scientific source of truth is the doctoral report:

    report1_25_08_2026.pdf

The report defines:

- model equations;
- event timing;
- state variables;
- behavioural rules;
- topology definitions;
- experiment design;
- metrics;
- mechanism diagnostics;
- endogenous-network extensions;
- stability analysis;
- state-space and estimation roadmap.

Legacy code is NOT a scientific specification.

Old code may be inspected for:

- implementation ideas;
- historical provenance;
- reproducibility of pilot results;
- useful numerical utilities.

However, old code must not override the equations or timing in the report.

In particular, the report explicitly states that the earlier pilot
implementation is not algebraically equivalent to the refined model,
including the refined executed-trade rule and price equation.

Therefore:

    old-result equivalence != refined-model correctness

---

## 3. Current Repository Architecture

The project has been restructured into the following main architecture:

    myproject/
    |
    |-- src/
    |   |-- model/
    |   |-- topologies/
    |   |-- experiments/
    |   |-- metrics/
    |   |-- analysis/
    |   `-- plotting/
    |
    |-- scripts/
    |-- slurm/
    |-- configs/
    |-- data/
    |-- results/
    |-- figures/
    |-- tests/
    |-- docs/
    `-- archive/legacy/

The intended data flow is:

    Configuration
        ->
    Experiment
        ->
    Topology + Parameters + Shock Path
        ->
    Simulation
        ->
    State / Flow Histories
        ->
    Metrics
        ->
    Raw Results
        ->
    Analysis
        ->
    Summary Tables / Figures

---

## 4. Current Legacy Architecture

The following modules represent the cleaned implementation of the earlier
pilot / legacy model:

    src/model/market_core.py
    src/model/baseline_env.py
    src/model/adaptive_env.py

Existing topology utilities are located in:

    src/topologies/

Existing experiment implementations include:

    src/experiments/baseline.py
    src/experiments/oat_runner.py
    src/experiments/oat_sampling.py
    src/experiments/interaction_runner.py
    src/experiments/interaction_sampling.py
    src/experiments/legacy_adaptive_common.py

Existing analysis and plotting code has been separated into:

    src/analysis/
    src/metrics/
    src/plotting/

These modules remain useful for reproducing historical results but are not
the implementation target for the refined doctoral model.

---

## 5. Archived Legacy Code

Historical scripts have been moved into:

    archive/legacy/

This includes:

- old baseline pipelines;
- old baseline post-processing;
- old adaptive-grid pipelines;
- old miscellaneous scripts;
- old RL / PPO / NetPPO / CausalProbe code;
- old SLURM workflows.

Do not develop new research code inside `archive/legacy/`.

The archive exists only for provenance and historical reference.

---

## 6. Restructuring Status

Project restructuring is COMPLETE.

The following regression/equivalence tests passed at the restructuring
checkpoint:

    tests/test_adaptive_equivalence.py
    tests/test_aggregate_results_equivalence.py
    tests/test_baseline_equivalence.py
    tests/test_baseline_experiment_equivalence.py
    tests/test_metrics_equivalence.py

All five tests passed.

At the restructuring checkpoint:

    git status --short

was clean.

Generated outputs and runtime artefacts are excluded through `.gitignore`.

The tag

    restructuring-complete

marks the recoverable state immediately before refined-model development.

---

## 7. Current Iridis Environment

At the beginning of a new Iridis session, use:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model
    git status

Important:

The Iridis Python module sets a system PYTHONPATH that can contaminate the
project virtual environment.

Therefore:

    unset PYTHONPATH

must be run after activating `.venv`.

Do not rely on the default Iridis Python installation.

The required Python version for this project is currently:

    Python 3.12.6

---

## 8. Editing Files on Iridis

`nano` has repeatedly segfaulted on this environment.

Prefer shell heredocs:

    cat > path/to/file.py <<'PY'
    ...
    PY

or:

    cat > path/to/file.md <<'EOF'
    ...
    EOF

Do not inspect binary files such as PNG or NPZ files using `head`, `cat`,
or `sed`.

Use `file` first if the file type is uncertain.

---

## 9. Refined Model Status

Implementation of the refined fixed-topology model has STARTED under:

    src/model/refined/

The package scaffold now exists:

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

Implemented so far:

- `RefinedParameters` for the homogeneous first-stage benchmark;
- parameter validation for the report-defined core restrictions;
- `RefinedState` for persistent state `(theta, b, x, p, R, W)`;
- `PeriodOutputs` for non-persistent within-period objects;
- binary feasible-graph validation for Equations (35)-(36);
- graph neighbourhood and degree construction;
- graph-supported row-stochastic attention validation for Equations (37)-(38);
- initial-state validation for Equation (40), including inventory bounds;
- focused tests in `tests/test_refined_state_and_parameters.py`.

The legacy modules remain unchanged and must not be mutated into the refined
model.

The remaining refined runtime equations have NOT yet been implemented.

Important verification note:

The new test file has been added to the repository, but this ChatGPT coding
session did not have shell access to the Iridis working tree. The tests must
therefore be run on Iridis after pulling the branch before this checkpoint is
labelled test-passing.

---

## 10. Immediate Next Step

NEXT STEP:

    Implement PeriodShocks and the refined fundamentals/signals block,
    Equations (42)-(46), with unit tests.

Required sequence:

1. Pull the latest `refined-model` branch on Iridis.
2. Run `tests/test_refined_state_and_parameters.py`.
3. Implement `PeriodShocks` in `src/model/refined/shocks.py`.
4. Implement `update_fundamental(...)` for Equation (42).
5. Implement `stationary_fundamental_variance(...)` for Equation (43).
6. Implement `fundamental_value(...)` for Equation (44).
7. Implement `private_signals(...)` for Equations (45)-(46).
8. Add deterministic unit tests for the entire block.

Do not implement a large Monte Carlo simulation yet.

The first computational milestone remains:

    one deterministic period of the refined model
    produces the exact transition implied by the report.

The second milestone is:

    a small deterministic multi-period simulation.

The third milestone is:

    the alpha = 0 network-null test.

Only after those tests pass should topology experiments be run.

---

## 11. Planned Development Sequence

The planned sequence is:

    Phase 1
    Refined fixed-topology core model
    Equations (35)-(82)

        ->

    Phase 2
    Refined binary topology generators
    G separated from W

        ->

    Phase 3
    Unit tests and deterministic integration tests

        ->

    Phase 4
    Paired fixed-topology Monte Carlo design

        ->

    Phase 5
    Refined market metrics and CID

        ->

    Phase 6
    Influence, overlap, action-covariance, and mechanism diagnostics

        ->

    Phase 7
    alpha / beta / gamma_R experiments and heterogeneity

        ->

    Phase 8
    Endogenous feasible-network formation and rewiring

        ->

    Phase 9
    Equilibrium, full Jacobian, spectral radius, and Lyapunov analysis

        ->

    Phase 10
    State-space representation, synthetic recovery, EKF, and empirical work

        ->

    Phase 11
    Planner / policy analysis

        ->

    Optional later extension
    MARL

---

## 12. Definition of a Successful Refined Implementation

A refined implementation is accepted only when:

- equations match the report;
- timing matches Equation (39);
- G and W are separate objects;
- W_t cannot affect b_t contemporaneously;
- inventory limits are respected;
- executed trade is distinguished from desired trade;
- profit uses the inherited position x_{t-1};
- the price equation contains the fundamental anchor;
- return conventions are internally consistent;
- alpha = 0 removes the network propagation channel;
- random seeds are explicitly separated by purpose;
- tests pass before large simulations are submitted.

Historical equivalence with pilot code is not an acceptance criterion.

---

## 13. New-Chat Handoff Prompt

When starting a new ChatGPT conversation, use:

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

    Current task:
    Continue the refined fixed-topology implementation from the recorded
    NEXT STEP.
