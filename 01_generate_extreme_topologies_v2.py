# build_extreme_topologies.py
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
import pandas as pd


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


def normalize_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < eps, 1.0, row_sums)
    return P / row_sums


def build_random_fixed_network(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        candidates = np.array([j for j in range(N) if j != i], dtype=int)
        nbrs = rng.choice(candidates, size=K, replace=False)
        w = rng.random(K)
        w = w / w.sum()
        P[i, nbrs] = w
    return P


def build_scale_free_extreme_network(
    N: int,
    K: int,
    rng: np.random.Generator,
    alpha: float = 2.2,
    hub_boost_count: int = 3,
    hub_boost_strength: float = 12.0,
) -> np.ndarray:
    """
    Extreme scale-free style:
    - preferential attachment with super-linear exponent alpha
    - a few early hubs get extra attractiveness
    """
    P = np.zeros((N, N), dtype=float)

    attractiveness = np.ones(N, dtype=float)
    hub_ids = np.arange(min(hub_boost_count, N))

    for i in range(N):
        probs = attractiveness.copy()

        if hub_ids.size > 0:
            probs[hub_ids] *= hub_boost_strength

        probs[i] = 0.0
        probs = np.power(probs, alpha)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            candidates = np.array([j for j in range(N) if j != i], dtype=int)
            nbrs = rng.choice(candidates, size=K, replace=False)
        else:
            probs = probs / probs_sum
            nbrs = rng.choice(np.arange(N), size=K, replace=False, p=probs)

        w = rng.random(K)
        w = w / w.sum()
        P[i, nbrs] = w

        attractiveness[nbrs] += 1.0

    return P


def build_small_world_clustered_network(
    N: int,
    K: int,
    rng: np.random.Generator,
    beta: float = 0.02,
) -> np.ndarray:
    """
    Strongly clustered directed small-world:
    - start from directed ring lattice
    - very low rewiring probability beta
    """
    P = np.zeros((N, N), dtype=float)

    for i in range(N):
        nbrs = [((i + 1 + k) % N) for k in range(K)]

        for kk in range(K):
            if rng.random() < beta:
                forbidden = set(nbrs)
                forbidden.add(i)
                candidates = [j for j in range(N) if j not in forbidden]
                if candidates:
                    nbrs[kk] = int(rng.choice(candidates))

        w = rng.random(K)
        w = w / w.sum()
        P[i, np.asarray(nbrs, dtype=int)] = w

    return P


def adjacency_from_P(P: np.ndarray, threshold: float = 1e-15) -> np.ndarray:
    A = (P > threshold).astype(int)
    np.fill_diagonal(A, 0)
    return A


def reciprocity_rate(A: np.ndarray) -> float:
    directed_edges = A.sum()
    if directed_edges == 0:
        return 0.0
    mutual = np.logical_and(A == 1, A.T == 1).sum()
    return float(mutual / directed_edges)


def local_clustering_directed_proxy(A: np.ndarray) -> float:
    """
    Simple undirected clustering proxy from directed adjacency.
    """
    U = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(U, 0)

    N = U.shape[0]
    vals = []
    for i in range(N):
        nbrs = np.where(U[i] > 0)[0]
        d = len(nbrs)
        if d < 2:
            vals.append(0.0)
            continue

        sub = U[np.ix_(nbrs, nbrs)]
        edges = sub.sum() / 2.0
        possible = d * (d - 1) / 2.0
        vals.append(float(edges / possible) if possible > 0 else 0.0)

    return float(np.mean(vals))


def bfs_distances(U: np.ndarray, start: int) -> np.ndarray:
    N = U.shape[0]
    dist = np.full(N, np.inf)
    dist[start] = 0.0
    queue = [start]
    head = 0

    while head < len(queue):
        u = queue[head]
        head += 1
        nbrs = np.where(U[u] > 0)[0]
        for v in nbrs:
            if not np.isfinite(dist[v]):
                dist[v] = dist[u] + 1.0
                queue.append(v)

    return dist


def avg_shortest_path_proxy(A: np.ndarray, max_sources: int = 25) -> float:
    U = ((A + A.T) > 0).astype(int)
    np.fill_diagonal(U, 0)

    N = U.shape[0]
    srcs = np.arange(N)[: min(N, max_sources)]
    dvals = []

    for s in srcs:
        dist = bfs_distances(U, int(s))
        finite = dist[np.isfinite(dist) & (dist > 0)]
        if finite.size > 0:
            dvals.extend(finite.tolist())

    if len(dvals) == 0:
        return float("inf")
    return float(np.mean(dvals))


def spectral_metrics(P: np.ndarray) -> tuple[float, float]:
    eigvals = np.linalg.eigvals(P)
    mags = np.sort(np.abs(eigvals))[::-1]
    spectral_radius = float(mags[0]) if mags.size > 0 else 0.0
    second_modulus = float(mags[1]) if mags.size > 1 else 0.0
    return spectral_radius, second_modulus


def row_entropy_mean(P: np.ndarray, eps: float = 1e-12) -> float:
    row_ent = -(P * np.log(P + eps)).sum(axis=1)
    return float(np.mean(row_ent))


def col_entropy_mean(P: np.ndarray, eps: float = 1e-12) -> float:
    col = P.sum(axis=0)
    s = col.sum()
    if s <= eps:
        return 0.0
    q = col / s
    return float(-(q * np.log(q + eps)).sum())


def topology_metrics(name: str, P: np.ndarray) -> dict:
    A = adjacency_from_P(P)
    indeg_w = P.sum(axis=0)

    sr, lam2 = spectral_metrics(P)

    top5_share = float(np.sort(indeg_w)[::-1][:5].sum() / max(indeg_w.sum(), 1e-12))

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
        "spectral_radius": sr,
        "second_eigenvalue_modulus": lam2,
        "row_entropy_mean": row_entropy_mean(P),
        "col_entropy_mean": col_entropy_mean(P),
    }


def save_topology(outdir: str, name: str, P: np.ndarray):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.npz")
    np.savez_compressed(path, P=P)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--beta_small_world", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="topologies_extreme")
    args = parser.parse_args()

    rng_random = np.random.default_rng(args.seed + 11)
    rng_sf = np.random.default_rng(args.seed + 22)
    rng_sw = np.random.default_rng(args.seed + 33)

    P_random = build_random_fixed_network(args.N, args.K, rng_random)
    P_sf = build_scale_free_extreme_network(args.N, args.K, rng_sf)
    P_sw = build_small_world_clustered_network(args.N, args.K, rng_sw, beta=args.beta_small_world)

    P_random = normalize_rows(P_random)
    P_sf = normalize_rows(P_sf)
    P_sw = normalize_rows(P_sw)

    save_topology(args.outdir, "random_fixed_extreme", P_random)
    save_topology(args.outdir, "scale_free_extreme", P_sf)
    save_topology(args.outdir, "small_world_clustered", P_sw)

    rows = [
        topology_metrics("random_fixed_extreme", P_random),
        topology_metrics("scale_free_extreme", P_sf),
        topology_metrics("small_world_clustered", P_sw),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "topology_structure_summary.csv"), index=False)

    print("Saved topologies to:", args.outdir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
