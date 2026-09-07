# D046 Exploratory Alpha Sweep Results

Status: VERIFIED EXPLORATORY RESULT

Date: 2026-09-07

## Provenance

Frozen D046 design:

    experiment seed = 2026090404
    alpha grid = (0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.95, 0.99)
    paired replications per alpha = 300
    topology triplet = (R, SW, SF)
    total simulations = 7200
    bootstrap seed = 2026090405
    bootstrap draws = 5000
    confidence level = 0.95

Execution:

    production Slurm array = 1526697
    48/48 tasks COMPLETED
    every task exit code = 0:0
    successful task markers = 48
    non-empty stderr files = 0
    complete alpha/replication checkpoints = 2400/2400
    production commit = 5283a338d75560f27d28a19d48e07f86cdeada07

Finalization:

    Slurm job = 1527139
    host = ruby047
    state = COMPLETED
    exit code = 0
    elapsed = 00:00:06
    memory = 8.17 MB
    commit = 5283a338d75560f27d28a19d48e07f86cdeada07

The finalizer verified the exact alpha-zero economic-path topology null across
the complete D046 sample.

## Main result

D046 separates a level effect from a topology-differentiation effect.

As alpha rises, the absolute level of several activity/instability outcomes
falls strongly across all topology classes. For example, return volatility
falls from about 0.00380 at alpha=0 to roughly 0.0020 near alpha=0.99, while
aggregate order-flow variance falls from about 111 to approximately 4--6.
Thus stronger social weight is not mechanically destabilising in levels under
the frozen D043/D044 calibration.

At the same time, cross-topology differentiation initially becomes stronger.
From alpha=0.20 through roughly alpha=0.85, the coherent ordering for return
volatility, mean absolute order flow, mean pairwise action covariance, and
aggregate order-flow variance is generally:

    SF > R > SW

The absolute aggregate order-flow-variance topology gap rises from about 0.159
at alpha=0.20 to about 2.061 at alpha=0.85. Return-volatility relative
differentiation rises from about 0.026% at alpha=0.20 to about 0.524% at
alpha=0.85.

The high-alpha region behaves differently. At alpha=0.95 and especially
alpha=0.99, several market-outcome rankings reverse while their absolute levels
continue to fall. At alpha=0.99, for example, the return-volatility ordering is
SW > SF > R rather than SF > R > SW. The same boundary region displays sharp
relative gaps partly because outcome levels have become small.

This region should therefore be described as a high-social-weight boundary
regime, not simply as stronger monotone topology amplification.

## Mechanism interpretation

The structural/influence ordering is much more stable than the downstream
market ordering. Realised hub influence remains approximately:

    SF >> R > SW

throughout the alpha grid. Attention overlap also retains a strong SF > R > SW
pattern over most of the grid.

Alpha therefore changes how strongly the existing influence architecture enters
belief formation and downstream action aggregation; it does not create the
structural hub hierarchy itself.

The D046 mechanism is consequently:

    topology-specific influence architecture
        -> social weight alpha activates that architecture in beliefs
        -> action covariance / aggregate flow differentiation rises
        -> realised price effects remain comparatively attenuated
        -> near alpha=1 the system enters a qualitatively different boundary regime

## Threshold outcome

The operational CID threshold is rarely exceeded. In the 300-replication D046
sample, threshold-exceeding rates are zero for all three topologies from
alpha=0.40 through alpha=0.99. D046 therefore does not support a claim that high
alpha alone generates severe operational instability.

## D047 anchor decision

D047 uses:

    alpha = 0.85

as its exploratory beta-sweep anchor.

This point is selected after D046, so D047 is explicitly exploratory rather
than confirmatory. Alpha=0.85 is preferred because it lies in the strongest
coherent pre-boundary differentiation region: the SF > R > SW mechanism remains
clear, absolute topology gaps are large relative to lower-alpha points, and the
ranking reversals observed at alpha=0.95--0.99 have not yet appeared.

The original D043/D045 alpha=0.75 remains the confirmatory baseline anchor and
is not replaced by this D047 design choice.
