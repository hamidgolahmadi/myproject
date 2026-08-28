import numpy as np


# =============================================================================
# Inequality / Concentration Metrics
# =============================================================================

def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the Gini coefficient of a non-negative vector.

    The Gini coefficient is used here as a measure of concentration or
    inequality in network influence. A value close to 0 indicates a relatively
    even distribution, while a value closer to 1 indicates strong
    concentration.

    Parameters
    ----------
    x : np.ndarray
        Input vector containing non-negative values.
    eps : float
        Small numerical tolerance used to avoid division by zero.

    Returns
    -------
    float
        Gini coefficient in the interval [0, 1].
    """

    # Convert the input to a floating-point NumPy array.
    x = np.asarray(x, dtype=float)

    # Negative values are not meaningful for this measure, so clip them to zero.
    x = np.clip(x, 0.0, None)

    # If the entire vector is effectively zero, inequality is defined as zero.
    if x.sum() <= eps:
        return 0.0

    # Sort values in ascending order, as required by the standard Gini formula.
    x = np.sort(x)

    n = len(x)
    index = np.arange(1, n + 1)

    # Standard discrete Gini coefficient formula.
    g = (
        2.0 * np.sum(index * x) / (n * np.sum(x))
        - (n + 1) / n
    )

    # Numerical rounding may produce tiny values outside [0, 1].
    return float(max(0.0, min(1.0, g)))


# =============================================================================
# Adjacency Construction
# =============================================================================

def adjacency_from_P(
    P: np.ndarray,
    threshold: float = 1e-15,
) -> np.ndarray:
    """
    Convert a weighted influence matrix into a binary adjacency matrix.

    Any entry greater than the threshold is treated as an active directed link.

    Parameters
    ----------
    P : np.ndarray
        Weighted influence matrix.
    threshold : float
        Minimum weight required for a link to be considered active.

    Returns
    -------
    np.ndarray
        Binary adjacency matrix.
    """

    # Convert positive influence weights into binary links.
    A = (P > threshold).astype(int)

    # Explicitly remove self-links.
    np.fill_diagonal(A, 0)

    return A


# =============================================================================
# Reciprocity
# =============================================================================

def reciprocity_rate(A: np.ndarray) -> float:
    """
    Measure the fraction of directed links that are reciprocated.

    If i -> j exists and j -> i also exists, the link is considered reciprocal.

    Parameters
    ----------
    A : np.ndarray
        Binary directed adjacency matrix.

    Returns
    -------
    float
        Reciprocity rate.
    """

    # Total number of directed edges.
    directed_edges = A.sum()

    # Avoid division by zero for an empty graph.
    if directed_edges == 0:
        return 0.0

    # Count directed edges that have a reverse counterpart.
    mutual = np.logical_and(A == 1, A.T == 1).sum()

    return float(mutual / directed_edges)


# =============================================================================
# Clustering
# =============================================================================

def local_clustering_directed_proxy(A: np.ndarray) -> float:
    """
    Compute an approximate clustering coefficient for a directed network.

    The directed adjacency matrix is first converted into an undirected proxy.
    Local clustering is then computed for each node and averaged across nodes.

    Parameters
    ----------
    A : np.ndarray
        Binary directed adjacency matrix.

    Returns
    -------
    float
        Mean local clustering coefficient.
    """

    # Convert the directed graph into an undirected proxy.
    U = ((A + A.T) > 0).astype(int)

    # Remove self-links.
    np.fill_diagonal(U, 0)

    N = U.shape[0]
    values = []

    for i in range(N):

        # Identify neighbours of node i.
        nbrs = np.where(U[i] > 0)[0]
        d = len(nbrs)

        # Clustering is zero when a node has fewer than two neighbours.
        if d < 2:
            values.append(0.0)
            continue

        # Extract the subgraph formed by node i's neighbours.
        sub = U[np.ix_(nbrs, nbrs)]

        # Each undirected edge appears twice in the adjacency matrix.
        edges = sub.sum() / 2.0

        # Maximum possible number of links among d neighbours.
        possible = d * (d - 1) / 2.0

        values.append(
            float(edges / possible)
            if possible > 0
            else 0.0
        )

    return float(np.mean(values))


# =============================================================================
# Shortest-Path Calculations
# =============================================================================

def bfs_distances(
    U: np.ndarray,
    start: int,
) -> np.ndarray:
    """
    Compute shortest-path distances from one source using breadth-first search.

    Parameters
    ----------
    U : np.ndarray
        Binary undirected adjacency matrix.
    start : int
        Index of the source node.

    Returns
    -------
    np.ndarray
        Vector of shortest-path distances from the source node.
        Unreachable nodes remain equal to infinity.
    """

    N = U.shape[0]

    # Initialise all distances as unreachable.
    dist = np.full(N, np.inf)

    # The source node is at distance zero from itself.
    dist[start] = 0.0

    # Breadth-first-search queue.
    queue = [start]
    head = 0

    while head < len(queue):

        # Pop the next node from the queue.
        u = queue[head]
        head += 1

        # Find neighbours of the current node.
        nbrs = np.where(U[u] > 0)[0]

        for v in nbrs:

            # Visit each node only once.
            if not np.isfinite(dist[v]):
                dist[v] = dist[u] + 1.0
                queue.append(v)

    return dist


def avg_shortest_path_proxy(
    A: np.ndarray,
    max_sources: int = 25,
) -> float:
    """
    Estimate the average shortest-path length of a network.

    To reduce computational cost, shortest paths are calculated only from a
    subset of source nodes when the network is large.

    Parameters
    ----------
    A : np.ndarray
        Binary directed adjacency matrix.
    max_sources : int
        Maximum number of source nodes used in the approximation.

    Returns
    -------
    float
        Approximate average shortest-path length.
    """

    # Convert the directed network into an undirected proxy.
    U = ((A + A.T) > 0).astype(int)

    # Remove self-links.
    np.fill_diagonal(U, 0)

    N = U.shape[0]

    # Use at most max_sources source nodes.
    sources = np.arange(N)[: min(N, max_sources)]

    distances = []

    for source in sources:

        # Compute shortest paths from the selected source node.
        dist = bfs_distances(U, int(source))

        # Keep finite and strictly positive distances only.
        finite = dist[
            np.isfinite(dist)
            & (dist > 0)
        ]

        distances.extend(finite.tolist())

    # Return infinity if no connected pair exists.
    if not distances:
        return float("inf")

    return float(np.mean(distances))


# =============================================================================
# Spectral Network Metrics
# =============================================================================

def spectral_metrics(
    P: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the spectral radius and second-largest eigenvalue modulus.

    For a row-stochastic matrix, the largest eigenvalue modulus is typically
    equal to one. The second-largest modulus is informative about the rate at
    which influence or information may mix across the network.

    Parameters
    ----------
    P : np.ndarray
        Weighted influence matrix.

    Returns
    -------
    tuple[float, float]
        Spectral radius and second-largest eigenvalue modulus.
    """

    # Compute all eigenvalues of the influence matrix.
    eigvals = np.linalg.eigvals(P)

    # Sort eigenvalue magnitudes from largest to smallest.
    magnitudes = np.sort(np.abs(eigvals))[::-1]

    spectral_radius = (
        float(magnitudes[0])
        if magnitudes.size > 0
        else 0.0
    )

    second_modulus = (
        float(magnitudes[1])
        if magnitudes.size > 1
        else 0.0
    )

    return spectral_radius, second_modulus


# =============================================================================
# Entropy-Based Metrics
# =============================================================================

def row_entropy_mean(
    P: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute the average entropy of the rows of the influence matrix.

    Higher row entropy indicates that agents distribute their influence weights
    more evenly across neighbours. Lower entropy indicates more concentrated
    individual attention.

    Parameters
    ----------
    P : np.ndarray
        Weighted influence matrix.
    eps : float
        Small numerical constant used inside the logarithm.

    Returns
    -------
    float
        Mean row entropy.
    """

    # Compute Shannon entropy independently for each row.
    row_entropy = -(
        P * np.log(P + eps)
    ).sum(axis=1)

    return float(np.mean(row_entropy))


def col_entropy_mean(
    P: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute entropy of total incoming influence across nodes.

    This measures how evenly realised incoming influence is distributed across
    the network.

    Parameters
    ----------
    P : np.ndarray
        Weighted influence matrix.
    eps : float
        Small numerical constant used for stability.

    Returns
    -------
    float
        Entropy of the normalised column influence distribution.
    """

    # Sum each column to obtain total incoming influence per node.
    col = P.sum(axis=0)

    total = col.sum()

    # Handle an effectively empty matrix safely.
    if total <= eps:
        return 0.0

    # Convert total incoming influence into a probability distribution.
    q = col / total

    # Compute Shannon entropy.
    return float(
        -(q * np.log(q + eps)).sum()
    )


# =============================================================================
# Combined Topology Summary
# =============================================================================

def topology_metrics(
    name: str,
    P: np.ndarray,
) -> dict:
    """
    Compute the main structural diagnostics for a weighted topology.

    Parameters
    ----------
    name : str
        Descriptive topology name.
    P : np.ndarray
        Weighted row-stochastic influence matrix.

    Returns
    -------
    dict
        Dictionary containing the main structural network statistics.
    """

    # Construct a binary adjacency representation.
    A = adjacency_from_P(P)

    # Column sums measure total incoming weighted influence.
    indeg_w = P.sum(axis=0)

    # Spectral diagnostics.
    spectral_radius, second_modulus = spectral_metrics(P)

    # Share of total incoming influence captured by the five largest nodes.
    top5_share = float(
        np.sort(indeg_w)[::-1][:5].sum()
        / max(indeg_w.sum(), 1e-12)
    )

    # Return all structural metrics in a single dictionary.
    return {
        "topology": name,
        "N": int(P.shape[0]),
        "K_out_mean": float(A.sum(axis=1).mean()),
        "indeg_mean": float(indeg_w.mean()),
        "indeg_std": float(indeg_w.std()),
        "indeg_max": float(indeg_w.max()),
        "indeg_gini": float(gini_coefficient(indeg_w)),
        "share_top5_indeg": top5_share,
        "reciprocity_rate": reciprocity_rate(A),
        "clustering_proxy": local_clustering_directed_proxy(A),
        "avg_shortest_path_proxy": avg_shortest_path_proxy(A),
        "spectral_radius": spectral_radius,
        "second_eigenvalue_modulus": second_modulus,
        "row_entropy_mean": row_entropy_mean(P),
        "col_entropy_mean": col_entropy_mean(P),
    }
