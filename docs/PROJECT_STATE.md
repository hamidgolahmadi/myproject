# Project State

Last updated: 2026-09-02

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

Refined experiment/evaluation layer:

    src/experiments/refined/
        seeding.py
        paired.py
        treatments.py
        structural.py
        calibration.py
        structural_io.py
        market_metrics.py
        action_covariance.py
        cid.py
        cid_events.py
        influence_metrics.py

Refined topology layer:

    src/topologies/refined/
        generators.py
        diagnostics.py

Structural-validation driver:

    scripts/run_refined_structural_validation.py

---

## 4. Binding Scientific Decisions

See `docs/DECISIONS.md`. Key frozen points include separation of `G` and `W_t`, lagged attention in beliefs, signed net order flow, inherited-position profit, paired common-random-number design, mandatory `alpha=0` negative control, and D041 structural-validation calibration.

No numerical CID threshold, component guardrail, rolling-window length, or CID reference scale has yet been frozen for the market experiment.

---

## 5. Verified Refined Core and Paired Infrastructure

The fixed-topology runtime for Equations (35)-(82) is implemented and VERIFIED, including canonical timing, deterministic multi-period simulation, shock generation, paired semantic seeds, topology-specific graph generation, neutral graph-supported `W_0`, and generated-treatment `alpha=0` controls.

---

## 6. Structural Validation — COMPLETED SUCCESSFULLY

D041:

    experiment_seed = 20260901
    N = 100
    K = 6
    n_graph_replications = 1000 per topology
    q = 5
    p_sw = 0.02
    a0 = 1.0

The 3000-graph structural run completed on Iridis.

Mean diagnostics:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

Structural separation is strong and non-overlapping in the intended dimensions. All symmetrised supports are fully connected. SW is highly clustered and fully connected with a limited-shortcut structure, but its APL-LCC is materially longer than Random; do not describe it as Random-like in path length.

---

## 7. Verified Test Checkpoints

Verified on Iridis:

    21 passed
    30 passed
    40 passed
    51 passed
    66 passed
    82 passed
    90 passed
    100 passed
    115 passed
    138 passed
    166 passed
    199 passed
    219 passed
    247 passed
    261 passed
    284 passed   + run-level market outcomes, Eqs. (236)-(238), (288)-(289)
    302 passed   + rolling action covariance / order-flow variance decomposition, Eqs. (239)-(240)
    341 passed   + rolling CID components, standardisation, and dimensionless CID, Eqs. (241)-(246)
    374 passed   + threshold/duration/right-censored stabilisation logic, Eqs. (247)-(250)

Latest verified checkpoint:

    374 passed in 6.75s

with clean working tree and branch up to date with `origin/refined-model`.

---

## 8. Market Outcome, Covariance, and CID Layers — VERIFIED

Verified modules:

    src/experiments/refined/market_metrics.py
    src/experiments/refined/action_covariance.py
    src/experiments/refined/cid.py
    src/experiments/refined/cid_events.py

Implemented and VERIFIED:

    Eq. (236) return volatility RV
    Eq. (237) RMS mispricing RMSM
    Eq. (237) maximum absolute mispricing MAM
    Eq. (238) mean absolute signed net order flow per agent MAF
    Eq. (239) rolling average pairwise sample action covariance
    Eq. (240) exact rolling signed-order-flow variance decomposition
    Eq. (241) rolling sample return volatility V_ret
    Eq. (242) rolling belief dispersion B_bel
    Eq. (243) RMS signed net-order-flow pressure Q_F
    Eq. (244) explicit positive reference scales
    Eq. (245) non-negative weights summing to one
    Eq. (246) dimensionless weighted CID
    Eq. (247) threshold-exceedance indicator with optional component guardrails
    Eq. (248) topology-level exceedance-rate aggregator
    Eq. (249) peak CID and CID-only exceedance-duration share
    Eq. (250) operational stabilisation with L_stab consecutive admissible windows and right-censoring
    Eq. (288) mean absolute return MAR
    Eq. (289) time-averaged cross-sectional belief variance V_b

Important threshold semantics are VERIFIED: Eq. (247) uses strict `>`; Eq. (250) uses `<=`; inactive component guardrails are treated as `+infinity`; right-censored runs are not assigned artificial stabilisation times. The report-defined first-stage default is `L_stab=50`.

No numerical `c_CID`, component guardrail, CID reference scale, or market rolling window is hard-coded.

---

## 9. Realised Influence / Common Exposure — NEW

NEWLY IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION:

    src/experiments/refined/influence_metrics.py

Public API:

    structural_hub_nodes(...)
    attention_entropy(...)
    normalised_attention_entropy(...)
    effective_number_of_sources(...)
    realised_influence_shares(...)
    realised_influence_hhi(...)
    realised_hub_influence_share(...)
    attention_overlap(...)
    attention_mobility(...)
    RealisedInfluencePoint
    RealisedInfluencePath
    realised_influence_path(...)

Implements:

    Eq. (251) row-level entropy normalised by log out-degree
    Eq. (252) effective number of sources exp(H)
    Eq. (253) network-average normalised entropy and effective-source count
    Eqs. (254)-(255) realised source influence shares from W_t column sums
    Eq. (256) realised-influence HHI
    Eq. (257) realised influence share of structural hubs H_q(G)
    Eqs. (258)-(264) aggregate attention overlap / common exposure
    Eq. (265) RMS row-level one-period attention mobility

The path evaluator records t=1,...,T. Period-t concentration/overlap uses `W_t = states[t].attention`; mobility uses `(W_{t-1}, W_t)`, so the first value measures the transition from neutral `W_0` to `W_1`.

Structural hubs are selected from directed in-degree in G, never from realised W_t. The report does not specify how to break an in-degree tie at the q-th rank; implementation uses decreasing in-degree then increasing node label as a deterministic reproducibility convention. This tie-break has no economic meaning.

For fixed-out-degree graphs under neutral uniform W_0, the implementation/test design checks the identity:

    realised source share = in-degree / total links

and therefore:

    S^I_q,0 = S^G_q

before adaptive attention reallocates influence.

The implementation stores agent-level entropy/effective-source arrays and source-level influence shares as well as network scalar summaries, so mechanism analysis does not require rerunning dynamics.

Eq. (266)-(267) KL divergence to a transition prior remains DEFERRED with the attention-inertia extension because the current first-stage model uses the frictionless `tau=0` attention rule.

New test file:

    tests/test_refined_influence_metrics.py

adds 35 pytest cases.

Expected next total:

    409 passed

---

## 10. Computational Milestones

Milestones 1-14 are VERIFIED, including structural validation, market outcomes, action covariance, CID construction, and threshold/stabilisation logic.

Milestone 15 — realised-influence concentration, overlap, and mobility, Eqs. (251)-(265):

    IMPLEMENTED; AWAITING IRIDIS TEST VERIFICATION

Do not start the large refined market Monte Carlo yet. After influence verification, the remaining critical gate is explicit market-evaluation calibration: rolling window `L`, CID reference scales, CID weights, `c_CID`, optional component guardrails, and `L_stab=50` must be frozen independently of topology rankings.

---

## 11. Immediate Next Step

1. Pull latest `refined-model` on Iridis.
2. Run:

       python -m pytest -q tests/test_refined_*.py

3. Expected total:

       409 passed

4. If all 409 pass, record Milestone 15 as VERIFIED.
5. Then design and freeze the market-evaluation calibration protocol before any topology-evaluation market Monte Carlo:

       rolling window L
       separate no-social calibration sample / seeds
       c_ret, c_bel, c_F
       CID weights
       c_CID
       optional component guardrails
       L_stab = 50

6. Calibration inputs must be fixed independently of observed R/SW/SF rankings.
7. After calibration is frozen, build the paired market-run driver and persistent run-level/mechanism output layer, then run a small smoke experiment before any 1000-replication confirmatory job.

---

## 12. Planned Development Sequence

    Phase 1  Refined fixed-topology core model                         COMPLETE
    Phase 2  Refined binary topology generators                       COMPLETE
    Phase 3  Deterministic integration and multi-period tests         COMPLETE
    Phase 4  Paired fixed-topology design + structural validation      COMPLETE
    Phase 5  Refined market metrics and CID                            NEAR COMPLETE
    Phase 6  Influence / overlap / action-covariance diagnostics       IN PROGRESS
    Phase 7  alpha / beta / gamma_R experiments and heterogeneity      PLANNED
    Phase 8  Endogenous feasible-network formation and rewiring        PLANNED
    Phase 9  Full equilibrium Jacobian / Lyapunov analysis             PLANNED
    Phase 10 State-space / synthetic recovery / EKF / empirical work   PLANNED
    Phase 11 Planner / policy analysis                                 PLANNED

Optional later extension: MARL

---

## 13. New-Chat Handoff Prompt

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
