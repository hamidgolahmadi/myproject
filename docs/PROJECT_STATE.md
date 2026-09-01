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

The refined model from the doctoral report has NOT yet been implemented.

The current legacy modules must not simply be edited until they appear to
match the report.

The refined implementation will be developed separately under:

    src/model/refined/

Proposed initial structure:

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

The first implementation target is ONLY the refined fixed-topology market
model defined by Equations (35)-(82).

---

## 10. Immediate Next Step

NEXT STEP:

    Implement the refined core market model, Equations (35)-(82).

Before implementation:

1. Read `docs/IMPLEMENTATION_MAP.md`.
2. Read `docs/DECISIONS.md`.
3. Verify the relevant equations directly against the doctoral report.
4. Create the `src/model/refined/` package.
5. Implement the model block-by-block.
6. Write tests alongside the implementation.

The first computational milestone is NOT a large Monte Carlo simulation.

The first milestone is:

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
    Implement the refined fixed-topology model, beginning with
    Equations (35)-(82).

