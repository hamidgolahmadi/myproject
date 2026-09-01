"""
Extreme topology generators.

This module contains deliberately pronounced network structures used in
experiments where we want the structural differences between topology classes
to be clearly identifiable.

The functions in this file are responsible only for generating topology
matrices. Network diagnostics, saving/loading, simulation, and plotting are
handled elsewhere in the project.
"""

import numpy as np


# =============================================================================
# Extreme Scale-Free Topology
# =============================================================================

def build_scale_free_extreme(
    n: int,
    k: int,
    rng: np.random.Generator,
    alpha: float = 2.2,
    hub_boost_count: int = 3,
    hub_boost_strength: float = 12.0,
) -> np.ndarray:
    """
    Generate an intentionally hub-dominated scale-free-style influence network.

    The network uses a preferential-attachment mechanism in which nodes that
    have already attracted influence become increasingly likely to attract
    additional links.

    The topology is made deliberately more concentrated through two mechanisms:

    1. Super-linear preferential attachment controlled by ``alpha``.
    2. Additional attractiveness assigned to a small number of potential hubs.

    This construction is useful when we want the scale-free topology to be
    structurally distinct from random and small-world benchmark networks.

    Parameters
    ----------
    n : int
        Number of nodes in the network.

    k : int
        Number of outgoing influence links assigned to each node.

    rng : np.random.Generator
        NumPy random-number generator. Passing the generator explicitly allows
        the calling experiment to control reproducibility.

    alpha : float, optional
        Exponent applied to node attractiveness. Values greater than one create
        stronger preferential attachment. Default is 2.2.

    hub_boost_count : int, optional
        Number of early nodes given additional attractiveness. Default is 3.

    hub_boost_strength : float, optional
        Multiplicative attractiveness boost applied to the designated hubs.
        Default is 12.0.

    Returns
    -------
    np.ndarray
        An ``n x n`` weighted influence matrix whose rows sum to one.
    """

    # Initialise the weighted influence matrix.
    P = np.zeros((n, n), dtype=float)

    # Every node begins with the same baseline attractiveness.
    attractiveness = np.ones(n, dtype=float)

    # Select the first few nodes as potential structural hubs.
    hub_ids = np.arange(
        min(hub_boost_count, n)
    )

    # Construct the outgoing influence links row by row.
    for i in range(n):

        # Start from the current attractiveness of all candidate nodes.
        probs = attractiveness.copy()

        # Give designated hubs additional attractiveness.
        if hub_ids.size > 0:
            probs[hub_ids] *= hub_boost_strength

        # Self-influence is not permitted.
        probs[i] = 0.0

        # Super-linear transformation strengthens preferential attachment.
        probs = np.power(
            probs,
            alpha,
        )

        probs_sum = probs.sum()

        # In the unlikely event that all probabilities collapse to zero,
        # fall back to uniform random selection among all other nodes.
        if probs_sum <= 0.0:

            candidates = np.array(
                [j for j in range(n) if j != i],
                dtype=int,
            )

            nbrs = rng.choice(
                candidates,
                size=k,
                replace=False,
            )

        else:

            # Convert attractiveness scores into valid selection probabilities.
            probs = probs / probs_sum

            # Select k distinct influence targets.
            nbrs = rng.choice(
                np.arange(n),
                size=k,
                replace=False,
                p=probs,
            )

        # Assign heterogeneous positive weights to the selected links.
        weights = rng.random(k)

        # Normalise the weights so that the row sums to one.
        weights = weights / weights.sum()

        # Store the weighted outgoing links.
        P[i, nbrs] = weights

        # Nodes that receive links become more attractive in subsequent steps.
        # This is the preferential-attachment mechanism.
        attractiveness[nbrs] += 1.0

    return P


# =============================================================================
# Strongly Clustered Small-World Topology
# =============================================================================

def build_small_world_clustered(
    n: int,
    k: int,
    rng: np.random.Generator,
    beta: float = 0.02,
) -> np.ndarray:
    """
    Generate a strongly clustered directed small-world influence network.

    The network begins as a directed ring lattice. Each node initially places
    influence on the next ``k`` nodes around the ring.

    Individual links are then rewired with a small probability ``beta``.
    Keeping ``beta`` low preserves strong local clustering while introducing
    occasional long-range connections.

    Parameters
    ----------
    n : int
        Number of nodes in the network.

    k : int
        Number of outgoing influence links assigned to each node.

    rng : np.random.Generator
        NumPy random-number generator used for rewiring and weight generation.

    beta : float, optional
        Probability that each lattice link is rewired. A small value produces
        strong clustering. Default is 0.02.

    Returns
    -------
    np.ndarray
        An ``n x n`` weighted influence matrix whose rows sum to one.
    """

    # Initialise the weighted influence matrix.
    P = np.zeros((n, n), dtype=float)

    # Construct the topology one row at a time.
    for i in range(n):

        # Begin with a directed ring lattice:
        # node i connects to the next k nodes around the ring.
        nbrs = [
            (i + 1 + offset) % n
            for offset in range(k)
        ]

        # Consider rewiring each local connection independently.
        for position in range(k):

            if rng.random() < beta:

                # Existing neighbours cannot be selected again.
                forbidden = set(nbrs)

                # Self-links are also prohibited.
                forbidden.add(i)

                # Identify valid alternative destinations.
                candidates = [
                    j
                    for j in range(n)
                    if j not in forbidden
                ]

                # Rewire only when at least one valid candidate exists.
                if candidates:
                    nbrs[position] = int(
                        rng.choice(candidates)
                    )

        # Generate heterogeneous influence weights.
        weights = rng.random(k)

        # Normalise the weights so that each row sums to one.
        weights = weights / weights.sum()

        # Store the weighted outgoing links.
        P[
            i,
            np.asarray(nbrs, dtype=int)
        ] = weights

    return P
