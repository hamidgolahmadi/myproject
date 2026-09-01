# scale_free_network.py
import numpy as np

def build_P_scale_free(N: int, K: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    attractiveness = np.ones(N, dtype=float)
    P = np.zeros((N, N), dtype=float)

    for i in range(N):
        probs = attractiveness.copy()
        probs[i] = 0.0
        probs = probs / probs.sum()

        nbrs = rng.choice(np.arange(N), size=K, replace=False, p=probs)
        w = rng.random(K)
        w = w / w.sum()
        P[i, nbrs] = w

        # preferential attachment
        attractiveness[nbrs] += 1.0

        # --- validate row ---
        P[i, i] = 0.0
        row_sum = P[i].sum()
        if row_sum <= 1e-12:
            raise RuntimeError(f"[scale_free] Row {i} is zero.")
        P[i] = P[i] / row_sum

    # --- global validate ---
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6), "[scale_free] Rows not stochastic"
    assert np.all(np.diag(P) == 0.0), "[scale_free] Self-loops detected"
    return P

