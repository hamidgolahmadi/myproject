# -*- coding: utf-8 -*-
"""
01_generate_extreme_topologies.py

Generate clearly distinct baseline topologies and save them to disk.

Outputs:
- topologies/*.npz
- results/topology_summary.csv

Usage:
    python 01_generate_extreme_topologies.py --n 80 --k 6 --seeds 20
"""

from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s < eps:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * (cum / (cum[-1] + eps)).sum()) / n
    return float(max(0.0, min(1.0, g)))


def row_stochastic_from_neighbors(
    n: int,
    neighbors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    p = np.zeros((n, n), dtype=float)
    for i in range(n):
        p[i, neighbors[i]] = weights[i]
    return p


# =============================================================================
# Topology builders
# =============================================================================

def build_random_fixed_network(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    neighbors = np.zeros((n, k), dtype=int)
    weights = np.zeros((n, k), dtype=float)

    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        nbrs = rng.choice(candidates, size=k, replace=False)
        w = rng.random(k)
        w /= w.sum()

        neighbors[i] = nbrs
        weights[i] = w

    return row_stochastic_from_neighbors(n, neighbors, weights)


def build_scale_free_extreme_network(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Deliberately more unequal than the earlier mild version.
    Preferential attachment with stronger hub bias.
    """
    neighbors = np.zeros((n, k), dtype=int)
    weights = np.zeros((n, k), dtype=float)

    attractiveness = np.ones(n, dtype=float)

    for i in range(n):
        probs = attractiveness.copy()
        probs[i] = 0.0
        probs = probs / probs.sum()

        nbrs = rng.choice(np.arange(n), size=k, replace=False, p=probs)

        w = rng.random(k)
        w /= w.sum()

        neighbors[i] = nbrs
        weights[i] = w

        attractiveness[nbrs] += 3.0

    return row_stochastic_from_neighbors(n, neighbors, weights)


def build_small_world_clustered_network(
    n: int,
    k: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Directed ring-lattice with low rewiring probability.
    Creates clustered/local structure with relatively low inequality.
    """
    neighbors = np.zeros((n, k), dtype=int)
    weights = np.zeros((n, k), dtype=float)

    for i in range(n):
        nbrs: List[int] = []
        for kk in range(1, k + 1):
            j = (i + kk) % n
            if rng.random() < beta:
                candidates = [x for x in range(n) if x != i and x not in nbrs]
                j = int(rng.choice(candidates))
            nbrs.append(j)

        w = rng.random(k)
        w /= w.sum()

        neighbors[i] = np.asarray(nbrs, dtype=int)
        weights[i] = w

    return row_stochastic_from_neighbors(n, neighbors, weights)


# =============================================================================
# Structural diagnostics
# =============================================================================

def binary_adjacency_from_p(p: np.ndarray) -> np.ndarray:
    return (p > 0.0).astype(int)


def local_clustering_directed_approx(a: np.ndarray) -> float:
    """
    Simple undirected approximation for clustering.
    Good enough for baseline diagnostics.
    """
    au = ((a + a.T) > 0).astype(int)
    np.fill_diagonal(au, 0)

    n = au.shape[0]
    vals = []

    for i in range(n):
        nbrs = np.where(au[i] > 0)[0]
        d = len(nbrs)
        if d < 2:
            vals.append(0.0)
            continue

        sub = au[np.ix_(nbrs, nbrs)]
        edges = sub.sum() / 2.0
        vals.append(float((2.0 * edges) / (d * (d - 1))))

    return float(np.mean(vals))


def average_shortest_path_undirected_approx(a: np.ndarray) -> float:
    """
    BFS on undirected approximation.
    """
    au = ((a + a.T) > 0).astype(int)
    np.fill_diagonal(au, 0)

    n = au.shape[0]
    dists_all = []

    for s in range(n):
        dist = np.full(n, np.inf)
        dist[s] = 0
        queue = [s]

        while queue:
            u = queue.pop(0)
            nbrs = np.where(au[u] > 0)[0]
            for v in nbrs:
                if np.isinf(dist[v]):
                    dist[v] = dist[u] + 1
                    queue.append(v)

        finite = dist[np.isfinite(dist) & (dist > 0)]
        if finite.size > 0:
            dists_all.extend(finite.tolist())

    if len(dists_all) == 0:
        return np.nan
    return float(np.mean(dists_all))


def topology_summary(topology: str, seed: int, p: np.ndarray) -> Dict[str, float]:
    indeg = p.sum(axis=0)
    a = binary_adjacency_from_p(p)

    return {
        "topology": topology,
        "seed": seed,
        "n": p.shape[0],
        "k_out_mean": float(a.sum(axis=1).mean()),
        "indeg_mean": float(indeg.mean()),
        "indeg_std": float(indeg.std()),
        "indeg_max": float(indeg.max()),
        "indeg_gini": float(gini_coefficient(indeg)),
        "p_sum": float(p.sum()),
        "p_sq_sum": float((p ** 2).sum()),
        "clustering_approx": float(local_clustering_directed_approx(a)),
        "avg_path_approx": float(average_shortest_path_undirected_approx(a)),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--beta_small_world", type=float, default=0.03)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out_topologies", type=str, default="topologies")
    parser.add_argument("--out_results", type=str, default="results")
    args = parser.parse_args()

    ensure_dir(args.out_topologies)
    ensure_dir(args.out_results)

    rows: List[Dict[str, float]] = []

    for seed in range(args.seeds):
        rng_r = np.random.default_rng(seed)
        rng_s = np.random.default_rng(seed)
        rng_w = np.random.default_rng(seed)

        p_random = build_random_fixed_network(args.n, args.k, rng_r)
        p_scale = build_scale_free_extreme_network(args.n, args.k, rng_s)
        p_small = build_small_world_clustered_network(
            args.n, args.k, args.beta_small_world, rng_w
        )

        items = [
            ("random_fixed", p_random),
            ("scale_free_extreme", p_scale),
            ("small_world_clustered", p_small),
        ]

        for topo_name, p in items:
            file_path = Path(args.out_topologies) / f"{topo_name}_seed_{seed}.npz"
            np.savez_compressed(file_path, P=p)

            rows.append(topology_summary(topo_name, seed, p))

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_results) / "topology_summary.csv", index=False)

    print("Saved topologies and topology_summary.csv")
    print(df.groupby("topology")[["indeg_gini", "clustering_approx", "avg_path_approx"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
