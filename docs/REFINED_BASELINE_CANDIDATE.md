# Archived Provisional Refined Baseline Candidate

Status: **ARCHIVED — SUPERSEDED BY D043**

The maintained first-stage specification is now documented in:

    docs/REFINED_BASELINE.md

This file is retained only to record the pre-freeze candidate history.

The provisional candidate matched the final D043 baseline except for:

    provisional sigma_0 = 1e-6
    frozen D043 sigma_0  = 5e-4

A topology-blind scale smoke showed finite, non-degenerate returns, mispricing,
order flow, desired actions, and inventory use, but also showed that
`sigma_0=1e-6` was roughly three orders of magnitude below typical local raw
reputation dispersion.

A common-random-number OAT sensitivity then compared:

    sigma_0 = {1e-6, 1e-4, 5e-4, 1e-3, 2e-3}

while holding graph realisations, shock paths, initial states, and every other
parameter fixed.  Market-scale outcomes were essentially invariant across the
grid, while attention mobility decreased smoothly as the regularisation floor
became economically relevant.

The final D043 choice `sigma_0=5e-4` was selected because it is of the same
order as realised local reputation dispersion, moderates the near-degenerate
standardisation problem targeted by Equation (58), and still leaves adaptive
attention active.  The choice used pooled absolute diagnostics only and did not
use topology rankings or confirmatory treatment effects.

For the full frozen vector, neutral initialisation rule, provenance, and smoke
statistics, use `docs/REFINED_BASELINE.md`.
