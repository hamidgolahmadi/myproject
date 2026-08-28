# random_fixed_network.py
import numpy as np

def build_P_random_fixed(N: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    P = np.zeros((N, N), dtype=float)

    for i in range(N):
        candidates = [j for j in range(N) if j != i]
        nbrs = rng.choice(candidates, size=K, replace=False)
        w = rng.random(K)
        w = w / w.sum()
        P[i, nbrs] = w

        # --- validate row ---
        P[i, i] = 0.0
        row_sum = P[i].sum()
        if row_sum <= 1e-12:
            raise RuntimeError(f"[random_fixed] Row {i} is zero.")
        P[i] = P[i] / row_sum

    # --- global validate ---
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "[random_fixed] Rows not stochastic"
    assert np.all(np.diag(P) == 0.0), "[random_fixed] Self-loops detected"
    return P
