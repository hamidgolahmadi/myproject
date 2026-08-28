# small_world_network.py
import numpy as np

def build_P_small_world(N: int, K: int, beta: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    P = np.zeros((N, N), dtype=float)

    for i in range(N):
        nbrs = []
        for k in range(1, K + 1):
            j = (i + k) % N  # ring lattice
            if rng.random() < beta:
                candidates = [x for x in range(N) if x != i and x not in nbrs]
                j = rng.choice(candidates)
            nbrs.append(j)

        w = rng.random(K)
        w = w / w.sum()
        P[i, nbrs] = w

        # --- validate row ---
        P[i, i] = 0.0
        row_sum = P[i].sum()
        if row_sum <= 1e-12:
            raise RuntimeError(f"[small_world] Row {i} is zero.")
        P[i] = P[i] / row_sum

    # --- global validate ---
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "[small_world] Rows not stochastic"
    assert np.all(np.diag(P) == 0.0), "[small_world] Self-loops detected"
    return P

