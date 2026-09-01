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

`unset PYTHONPATH` remains mandatory because the Iridis Python module injects a system PYTHONPATH.

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

Refined experiment/evaluation layer:

    src/experiments/refined/
        __init__.py
        seeding.py
        paired.py
        treatments.py
        structural.py
        calibration.py
        structural_io.py
        market_metrics.py

Refined topology layer:

    src/topologies/refined/
        __init__.py
        generators.py
        diagnostics.py

Structural-validation driver:

    scripts/run_refined_structural_validation.py

---

## 4. Binding Scientific Decisions

See `docs/DECISIONS.md` for the full register. Key frozen points:

- `G` is the directed binary feasible-information graph; `W_t` is separate graph-supported effective attention.
- Current beliefs use inherited `W_{t-1}`.
- Desired and executed trades are distinct.
- `F_t` is signed net order flow.
- Price includes fundamental correction and order-flow impact.
- Profit uses inherited position.
- First adaptive attention is frictionless reputation-sensitive softmax.
- `alpha=0` is the mandatory network-propagation negative control.
- Paired topology treatments share shocks and non-network initial conditions while graph randomness is topology-specific.
- Formal stability later uses the complete equilibrium Jacobian, not the spectral radius of `W`.
- D041 freezes the first structural-validation calibration.

---

## 5. Verified Refined Core

The fixed-topology runtime for Equations (35)-(82) is implemented and VERIFIED, including parameters/state, fundamentals/signals, lagged beliefs, adaptive attention, bounded/inventory-constrained trading, order flow, price, return, profit, reputation, the canonical one-period transition, deterministic multi-period simulation, and explicit shock-path generation.

The simulator contains no duplicate economic equations; it repeatedly calls `transition_one_period(...)`.

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

Semantic seed derivation lives in `src/experiments/refined/seeding.py`.

The generated R/SW/SF `alpha=0` end-to-end negative control is VERIFIED: graph and attention objects may differ, but all downstream economic paths coincide.

---

## 7. Verified Topology Generators and Diagnostics

Implemented and VERIFIED:

    generate_random_fixed_out_degree(...)
    generate_small_world(...)
    generate_hub_dominated(...)

All benchmark graphs are directed, simple, zero-diagonal, exactly `K` outgoing links per row, and `N*K` total links.

The hub-dominated generator uses attachment proportional to `in_degree_j + a0` and applies Equation (212) post-formation node relabelling.

Structural diagnostics in `src/topologies/refined/diagnostics.py` implement Equations (203)-(211): in-degree Gini, top-q hub link share, diagnostic-only symmetrised clustering, largest-component path length, and largest-component share.

---

## 8. Structural-Only Ensemble Validation — COMPLETED

D041 calibration:

    experiment_seed = 20260901
    N = 100
    K = 6
    n_graph_replications = 1000 per topology
    q = 5
    p_sw = 0.02
    a0 = 1.0

Canonical run:

    python scripts/run_refined_structural_validation.py

COMPLETED on Iridis with 3000 graph records and clean working tree.

Output files:

    results/refined/structural_validation/structural_graph_records.csv
    results/refined/structural_validation/structural_summary.csv
    results/refined/structural_validation/structural_metadata.json

Mean diagnostics:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

The structural treatment separation is SUCCESSFUL and is not merely a mean effect:

- SF Gini range: 0.42407 to 0.60907;
- Random Gini range: 0.16620 to 0.26670;
- therefore every realised SF graph in this ensemble has higher in-degree Gini than every realised Random graph.
- SF top-five hub-share range: 0.13667 to 0.28333;
- Random top-five hub-share range: 0.07833 to 0.11000;
- therefore every realised SF graph also has higher top-five structural concentration than every realised Random graph.
- SW clustering range: 0.50588 to 0.59055;
- Random clustering range: 0.08869 to 0.13089;
- therefore every realised SW graph has substantially higher clustering than every realised Random graph.
- all 3000 symmetrised supports have largest-component share 1.0.

Interpretation of SW path length must remain precise. Under `p_sw=0.02`, SW has materially higher clustering and remains fully connected, but its mean APL-LCC (4.47) is longer than the matched Random benchmark (2.10). Do not describe SW as having Random-like path length. Its treatment is strong local clustering plus a limited shortcut structure.

No recalibration is currently required merely to exaggerate separation: the intended concentration and clustering treatments are already strongly identified structurally.

---

## 9. Verified Test Checkpoints

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
    261 passed  + frozen structural calibration + output persistence + CLI smoke

Latest verified checkpoint:

    261 passed in 4.75s

with a clean working tree and branch up to date with `origin/refined-model`.

---

## 10. Run-Level Market Outcome Metrics — NEW

NEWLY IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION:

    src/experiments/refined/market_metrics.py

Implements the report evaluation sample and principal run-level outcomes from Section 5.5:

    Equation (236)  return volatility RV
    Equation (237)  RMS mispricing RMSM
    Equation (237)  maximum absolute mispricing MAM
    Equation (238)  mean absolute signed net order flow per agent MAF
    Equation (288)  mean absolute return MAR
    Equation (289)  time-averaged cross-sectional belief variance V_b

Public API:

    return_volatility(...)
    rms_mispricing(...)
    maximum_absolute_mispricing(...)
    mean_absolute_order_flow_per_agent(...)
    mean_absolute_return(...)
    time_averaged_belief_variance(...)
    compute_run_level_market_outcomes(...)
    RunLevelMarketOutcomes

Evaluation convention:

    T_B = {B+1,...,T}

is implemented by selecting `period_outputs[B:T]` and matching `states[B+1:T+1]`. Baseline uses `B=0`; positive burn-in remains an explicit robustness choice.

The metric layer evaluates `SimulationResult` only. It does not duplicate or rerun economic dynamics.

The direct tests explicitly distinguish:

    p_t - v_t        from inherited-price mispricing
    |F_t| / N        from gross trading volume sum_i |a_i,t|
    sample return SD from population SD
    population cross-sectional belief variance from sample variance

New test file:

    tests/test_refined_market_metrics.py

It adds 23 pytest cases.

Expected next total:

    284 passed

---

## 11. Computational Milestones

Milestones 1-10 are VERIFIED:

1. deterministic one-period transition
2. deterministic multi-period simulator
3. multi-period alpha=0 network-null test
4. explicit shock-path generation / CRN
5. semantic paired replication plan
6. refined benchmark topology generators + Eq. (212)
7. paired treatment construction + generated alpha=0 control
8. structural graph diagnostics, Eqs. (203)-(211)
9. structural-only matched ensemble runner
10. frozen structural calibration + persistent output driver

Structural ensemble validation itself is also COMPLETED SUCCESSFULLY.

Milestone 11 — principal run-level market metrics:

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

Do not start the large refined market Monte Carlo yet. Rolling action covariance, CID, effective-influence diagnostics, and explicit market-run calibration still need implementation/verification.

---

## 12. Immediate Next Step

1. Pull latest `refined-model` on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       284 passed

4. If all 284 pass, record Milestone 11 as VERIFIED.
5. Then implement the next Section 5.5 evaluation blocks in small stages:

       rolling action covariance and exact order-flow variance decomposition, Eqs. (239)-(240)
       rolling CID components and normalisation, Eqs. (241)-(246)
       threshold-exceedance / duration / censored stabilisation logic, Eqs. (247)-(250)
       effective-influence concentration / overlap / mobility, Eqs. (251)-(265)

6. Fix all CID reference scales, rolling-window lengths, thresholds, guardrails, and stabilisation length before topology-evaluation Monte Carlo. Do not reverse-engineer thresholds from desired topology rankings.

---

## 13. Planned Development Sequence

    Phase 1  Refined fixed-topology core model                         COMPLETE
    Phase 2  Refined binary topology generators                       COMPLETE
    Phase 3  Deterministic integration and multi-period tests         COMPLETE
    Phase 4  Paired fixed-topology design + structural validation      COMPLETE
    Phase 5  Refined market metrics and CID                            IN PROGRESS
    Phase 6  Influence / overlap / action-covariance diagnostics       PLANNED
    Phase 7  alpha / beta / gamma_R experiments and heterogeneity      PLANNED
    Phase 8  Endogenous feasible-network formation and rewiring        PLANNED
    Phase 9  Full equilibrium Jacobian / Lyapunov analysis             PLANNED
    Phase 10 State-space / synthetic recovery / EKF / empirical work   PLANNED
    Phase 11 Planner / policy analysis                                 PLANNED

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
