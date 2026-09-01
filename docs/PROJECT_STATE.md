# Project State

Last updated: 2026-09-01

## 1. Project Identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Primary branch:

    refined-model

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility only and must not override the report.

---

## 2. Iridis Session Setup

At the beginning of an Iridis session:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model
    git status

`unset PYTHONPATH` remains mandatory because the Iridis Python module injects a
system PYTHONPATH.

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
        calibration.py
        structural_io.py

Refined topology layer:

    src/topologies/refined/
        __init__.py
        generators.py
        diagnostics.py

Refined structural-validation driver:

    scripts/run_refined_structural_validation.py

---

## 4. Binding Scientific Decisions

See `docs/DECISIONS.md` for the full register. Important frozen points include:

- `G` is the directed binary feasible-information graph; `W_t` is separate
  graph-supported effective attention.
- current beliefs use inherited `W_{t-1}`;
- desired and executed trades are distinct;
- `F_t` is signed net order flow;
- price includes fundamental correction and order-flow impact;
- profit uses inherited position;
- first adaptive attention is frictionless reputation-sensitive softmax;
- `alpha=0` is the mandatory network-propagation negative control;
- paired topology treatments share shocks and non-network initial conditions,
  while graph randomness is topology-specific;
- formal stability later uses the complete equilibrium Jacobian, not the
  spectral radius of `W`.

New Decision D041 fixes the first structural-validation calibration.

---

## 5. Verified Refined Core

The fixed-topology runtime for Equations (35)-(82) is implemented and VERIFIED:

- parameters/state;
- graph/attention validation;
- fundamentals/signals;
- lagged belief updating;
- adaptive attention;
- valuation/trading/inventory/order flow;
- price/return/profit/reputation;
- canonical one-period transition;
- deterministic multi-period simulation;
- explicit shock-path generation with separated RNG streams.

The simulator contains no duplicate economic equations; it repeatedly calls
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

Semantic seed derivation lives in:

    src/experiments/refined/seeding.py

The generated R/SW/SF `alpha=0` end-to-end negative control is VERIFIED: graph
and attention objects may differ, but all downstream economic paths coincide.

---

## 7. Verified Topology Generators

Implemented and VERIFIED:

    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

All benchmark graphs are directed, simple, zero-diagonal, exactly `K`
outgoing links per row, and `N*K` total links.

The hub-dominated generator uses attachment proportional to
`in_degree_j + a0` and applies Equation (212) post-formation node relabelling.

---

## 8. Verified Structural Diagnostics

Implemented and VERIFIED under `src/topologies/refined/diagnostics.py`:

    in_degrees(G)
    in_degree_gini(G)
    hub_link_share(G, q)
    symmetrised_support(G)
    global_clustering(G)
    largest_component_share(G)
    average_path_length_lcc(G)
    diagnose_graph(G, q=q)
    diagnose_ensemble(graphs, q=q)

These follow Section 5.3.1, Equations (203)-(211). Concentration measures use
directed `G`; clustering/path/component measures use diagnostic-only
symmetrised support `G^u`.

---

## 9. Verified Structural-Only Ensemble Runner

Implemented and VERIFIED in:

    src/experiments/refined/structural.py

Objects/functions:

    StructuralEnsembleRecord
    DistributionSummary
    TopologyStructuralSummary
    StructuralEnsembleResult
    run_structural_ensemble(...)

The runner:

- generates graph ensembles only;
- uses the same semantic graph-seed mapping as later paired market runs;
- preserves graph-level records;
- computes Section 5.3.1 diagnostics;
- supplies descriptive summaries without discarding raw observations;
- does not generate shocks, market states, `W_t`, or market outcomes.

This layer was verified at the 247-test Iridis checkpoint.

---

## 10. First Structural-Validation Calibration — D041

FROZEN FOR THE FIRST STRUCTURAL VALIDATION RUN:

    experiment_seed = 20260901
    N = 100
    K = 6
    n_graph_replications = 1000
    q = 5
    p_sw = 0.02
    a0 = 1.0

Provenance:

- `N=100`, `K=6`, and 1000 graph realisations per topology match the report's
  retained first-stage baseline scale.
- `q=5` is the top-five / five-percent hub-share convention used in the report
  worked example and legacy structural diagnostic.
- `p_sw=0.02` comes from the legacy extreme structural-validation pipeline;
  the report specifies a small positive rewiring probability but no unique
  numerical calibration.
- `a0=1.0` is consistent with the report's illustration and with the legacy
  unit initial-attractiveness linear preferential-attachment baseline; the
  report itself requires only `a0>0`.
- the experiment seed is a reproducibility namespace only.

These are explicit calibration choices, not model equations. If the intended
ensemble separation fails, recalibration must be documented rather than
silently tuned.

The canonical code object is:

    StructuralValidationCalibration
    first_structural_validation_calibration()

in:

    src/experiments/refined/calibration.py

---

## 11. Structural Output Persistence and Driver

NEWLY IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION.

Output writer:

    src/experiments/refined/structural_io.py
    write_structural_result(...)

It writes:

    structural_graph_records.csv
    structural_summary.csv
    structural_metadata.json

The raw graph-level CSV is primary; the summary CSV is descriptive only.

Driver:

    scripts/run_refined_structural_validation.py

With no scientific overrides it executes D041. A reduced `--n-replications`
argument exists only for smoke/testing. The default output directory is:

    results/refined/structural_validation/

The real D041 run command will be:

    python scripts/run_refined_structural_validation.py

but only after the new calibration/output/CLI tests pass on Iridis.

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
    219 passed  + report-defined structural diagnostics
    247 passed  + structural-only ensemble runner + shared seed mapping

Latest verified checkpoint:

    247 passed in 2.29s

with a clean working tree and branch up to date with `origin/refined-model`.

New calibration/output/CLI tests:

    14 test cases

Expected next refined total:

    261 passed

---

## 13. Computational Milestones

Milestone 1 — deterministic one-period transition: VERIFIED

Milestone 2 — deterministic multi-period simulator: VERIFIED

Milestone 3 — multi-period alpha=0 network-null test: VERIFIED

Milestone 4 — explicit shock-path generation / CRN: VERIFIED

Milestone 5 — semantic paired replication plan: VERIFIED

Milestone 6 — refined benchmark topology generators + Eq. (212): VERIFIED

Milestone 7 — paired treatment construction + generated alpha=0 control: VERIFIED

Milestone 8 — structural graph diagnostics, Eqs. (203)-(211): VERIFIED

Milestone 9 — structural-only matched ensemble runner: VERIFIED

Milestone 10 — frozen structural calibration + persistent output driver:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

No large refined market Monte Carlo should be submitted yet.

---

## 14. Immediate Next Step

1. Pull latest `refined-model` on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       261 passed

4. If all 261 pass, record Milestone 10 as VERIFIED.
5. Then run the D041 structural-only ensemble validation, preferably as a
   compute-node/batch job rather than treating a 1000x3 ensemble as a unit test.
6. Inspect the full distributions and summaries for:

       in-degree Gini
       top-five hub link share
       global clustering
       average path length on the LCC
       largest-component share

7. Required qualitative validation before market interpretation:

       SF concentration > matched Random concentration, materially
       SW clustering > matched Random clustering, materially
       SW retains relatively short paths
       component coverage is reported alongside path length

The report supplies qualitative ensemble expectations, not universal numeric
cutoffs. Do not invent a hard pass/fail threshold.

---

## 15. Planned Development Sequence

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
