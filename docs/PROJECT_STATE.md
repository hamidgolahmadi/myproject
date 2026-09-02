# Project State

Last updated: 2026-09-02

## Project identity

Project root:

    /iridisfs/home/hg2e25/projects/myproject

Branch:

    refined-model

Scientific source of truth:

    report1_25_08_2026.pdf

Legacy code is reference/reproducibility only.

Iridis setup:

    cd /iridisfs/home/hg2e25/projects/myproject
    module load python/3.12.6
    source .venv/bin/activate
    unset PYTHONPATH
    git switch refined-model

## Verified architecture

Core refined model Eqs. (35)-(82): VERIFIED.

Paired semantic seeds, common shock paths, topology-specific G/W0, generated-treatment alpha=0 control: VERIFIED.

Structural diagnostics and D041 3000-graph validation: VERIFIED.

Structural-validation means:

              Gini      top-5 share   clustering    APL-LCC    LCC share
    R       0.21951       0.09371       0.10846      2.09894      1.00000
    SW      0.03201       0.05932       0.54982      4.46807      1.00000
    SF      0.51145       0.18908       0.13959      2.08013      1.00000

Evaluation/mechanism layers VERIFIED:

- Eqs. (236)-(238), (288)-(289): run-level outcomes;
- Eqs. (239)-(240): rolling action covariance and exact net-flow variance decomposition;
- Eqs. (241)-(246): CID components, scales, weights, CID;
- Eqs. (247)-(250): exceedance, duration, stabilisation, right-censoring;
- Eqs. (251)-(265): attention entropy/effective sources, realised influence shares/HHI, structural-hub realised influence, overlap, mobility.

Eq. (266)-(267) KL-to-transition-prior remains deferred with attention inertia.

## Latest verified test checkpoint

    504 passed in 5.88s

with clean working tree and branch up to date with origin/refined-model.

This verifies the D042 calibration method, provisional baseline object/initialisation, and baseline scale-smoke evaluator/driver.

## D042 market-evaluation calibration method — FROZEN

    T = 1000
    B = 0
    rolling L = 50
    robustness L = {25, 100}
    alpha_calibration = 0
    scale sample = 500 runs, namespace 2026090201
    threshold sample = 500 separate runs, namespace 2026090202
    reference scales = pooled component medians
    CID weights = equal thirds
    c_CID = 95th percentile of run-level peak CID, quantile method higher
    component guardrails = inactive baseline
    L_stab = 50

DO NOT run the 500+500 calibration yet.

## Provisional refined baseline candidate

Module:

    src/experiments/refined/baseline_specification.py

Provenance and smoke record:

    docs/REFINED_BASELINE_CANDIDATE.md

Status:

    PROVISIONAL — NOT FROZEN

Candidate parameters:

    rho_theta    = 0.985
    sigma_theta  = 0.025
    v_bar        = 0.0
    psi          = 1.0
    sigma_s      = 0.06
    sigma_b      = 0.025
    alpha        = 0.75
    kappa        = 2.4
    x_bar        = 5.0
    chi          = 0.02
    lambda_price = 0.0002
    sigma_p      = 0.001
    gamma_R      = 0.9
    beta         = 1.0
    sigma_0      = 1e-6   # under pre-freeze review

Neutral non-network initialisation candidate:

    theta_0 ~ stationary AR(1)
    b_i,0 = theta_0 for all i
    p_0 = v_bar + psi theta_0
    x_0 = 0
    R_0 = 0

W_0 remains topology-specific uniform graph-supported attention.

## Completed baseline scale smoke

Driver:

    scripts/run_refined_baseline_scale_smoke.py

Design:

    experiment_seed = 2026090203
    paired replications = 5
    total treatment runs = 15
    N = 100
    T = 1000

Pooled medians:

    return_std                              0.00342286
    mean_abs_return                         0.00271065
    max_abs_return                          0.0109005
    rms_mispricing                          0.0685422
    max_abs_mispricing                      0.219996
    mean_abs_flow_per_agent                 0.0784241
    rms_flow_per_agent                      0.101498
    desired_action_abs_p95                  0.304871
    desired_action_saturation_fraction      0.0
    execution_projection_fraction           0.14608
    inventory_boundary_fraction             0.14608
    median_local_reputation_std              0.000672808
    median_reputation_scale_to_sigma0        672.808
    mean_attention_mobility                  0.0492363
    max_attention_mobility                   0.664115
    final_attention_distance_from_initial    0.379522

Interpretation:

- returns, mispricing and signed flow are finite and non-degenerate;
- desired actions are not tanh-saturated;
- inventory constraints are active but not mechanically dominant;
- the unresolved issue is sigma_0: realised local reputation dispersion is hundreds of times larger than the provisional floor, making regularisation almost immediately negligible.

The report states that sigma_0 exists to prevent an almost-degenerate local reputation distribution from generating an artificial response after standardisation. Therefore sigma_0 must be checked before baseline freeze.

## sigma_0 pre-freeze sensitivity smoke — NEW

Module:

    src/experiments/refined/sigma0_sensitivity.py

Driver:

    scripts/run_refined_sigma0_sensitivity_smoke.py

Tests:

    tests/test_refined_sigma0_sensitivity.py

Controlled OAT grid:

    sigma_0 in {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

Design rules:

- reuse the same experiment seed 2026090203 and the same five replication ids;
- therefore graph seeds, shock paths and neutral initial-state randomness are common across sigma_0 values;
- change only sigma_0;
- pool diagnostics across topology labels;
- do not estimate R/SW/SF contrasts and do not tune toward a preferred topology ranking.

Reported pooled diagnostics focus on reputation-scale/sigma_0, attention mobility/distance, returns, mispricing, signed flow and inventory projection.

New tests add 24 pytest cases.

Expected next checkpoint:

    528 passed

## Immediate next step

1. Pull latest refined-model on Iridis.
2. Run all refined tests; expected 528 passed.
3. If green, run:

       python scripts/run_refined_sigma0_sensitivity_smoke.py

4. Inspect the pooled table only for scale/regularisation behavior.
5. Select sigma_0 on pre-freeze regularisation grounds, document the choice, and freeze the complete baseline vector only if the resulting model remains non-degenerate.
6. Recheck the alpha=0 topology-null property under the selected frozen vector.
7. Then run a small no-social D042 calibration smoke before the full 500+500 calibration.
8. Persist and freeze numerical c_ret, c_bel, c_F, and c_CID before any confirmatory topology market experiment.

Large topology-evaluation Monte Carlo remains prohibited until these gates pass.

## Development status

    Phase 1  Refined fixed-topology core                         COMPLETE
    Phase 2  Topology generators                                COMPLETE
    Phase 3  Deterministic integration                          COMPLETE
    Phase 4  Paired design + structural validation              COMPLETE
    Phase 5  Market metrics / CID / calibration method          COMPLETE
    Phase 6  Mechanism diagnostics                              COMPLETE
    Phase 7  Baseline scale smoke + market calibration          IN PROGRESS
    Phase 8  Paired confirmatory market runner                  PLANNED
    Phase 9  alpha/beta/gamma experiments + heterogeneity       PLANNED
    Phase 10 Endogenous G formation                             PLANNED
    Phase 11 Full Jacobian / Lyapunov                           PLANNED
    Phase 12 State-space / EKF / empirical work                PLANNED
    Phase 13 Planner / policy                                   PLANNED
