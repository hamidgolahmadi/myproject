# run_baselines.py
import argparse
import numpy as np

from random_fixed_network import build_P_random_fixed
from scale_free_network import build_P_scale_free
from small_world_network import build_P_small_world

# ===== metrics =====
def gini(x):
    x = np.abs(x)
    if x.sum() < 1e-12:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return (n + 1 - 2 * np.sum(np.cumsum(x) / np.sum(x))) / n

def P_signature(P):
    indeg = P.sum(axis=0)
    return dict(
        sumP=float(P.sum()),
        sumP2=float((P**2).sum()),
        indeg_mean=float(indeg.mean()),
        indeg_std=float(indeg.std()),
        indeg_max=float(indeg.max()),
        gini_indeg=float(gini(indeg)),
        indeg_first10=np.round(indeg[:10], 4).tolist(),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, required=True,
                        choices=["random", "scale_free", "small_world"])
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.topology == "random":
        P = build_P_random_fixed(args.N, args.K, seed=args.seed)
    elif args.topology == "scale_free":
        P = build_P_scale_free(args.N, args.K, seed=args.seed)
    elif args.topology == "small_world":
        P = build_P_small_world(args.N, args.K, beta=args.beta, seed=args.seed)
    else:
        raise ValueError("Unknown topology")

    sig = P_signature(P)

    print("=" * 80)
    print(f"TOPOLOGY = {args.topology}")
    print(f"N={args.N} K={args.K} beta={args.beta} seed={args.seed}\n")

    print(f"[{args.topology}] P signature:")
    print("  sum(P), sum(P^2), indeg_mean, indeg_std, indeg_max, gini(indeg)")
    print(f"  {sig['sumP']:.6f}  {sig['sumP2']:.6f}  {sig['indeg_mean']:.6f}  "
          f"{sig['indeg_std']:.6f}  {sig['indeg_max']:.6f}  {sig['gini_indeg']:.6f}")
    print(f"[{args.topology}] indeg first10: {sig['indeg_first10']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
