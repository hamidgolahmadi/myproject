# generate_abg_grid_samples.py

import os
import numpy as np
import pandas as pd


def main():
    # -----------------------------
    # Fixed baseline parameters
    # -----------------------------
    baseline = {
        "sigma_signal": 0.06,
        "sigma_belief": 0.025,
        "rho_y": 0.985,
        "sigma_y": 0.025,
        "trade_sensitivity": 2.4,
        "price_impact": 0.02,
        "sigma_price": 0.10,
    }

    # -----------------------------
    # Grid specification
    # -----------------------------
    alpha_values = np.array(
        [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00],
        dtype=float,
    )

    beta_values = np.geomspace(0.5, 50.0, 25)
    gamma_values = np.geomspace(0.01, 1.0, 25)

    # -----------------------------
    # Build full grid
    # -----------------------------
    rows = []
    sample_id = 0

    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                row = {
                    "sample_id": sample_id,
                    "alpha_social": float(alpha),
                    "beta": float(beta),
                    "gamma": float(gamma),
                    **baseline,
                }
                rows.append(row)
                sample_id += 1

    df = pd.DataFrame(rows)

    # -----------------------------
    # Save
    # -----------------------------
    outdir = "interaction_parameter_samples"
    os.makedirs(outdir, exist_ok=True)

    outpath = os.path.join(outdir, "alpha_beta_gamma_grid.csv")
    df.to_csv(outpath, index=False)

    print(f"Saved {outpath}")
    print(f"Total samples: {len(df)}")
    print(f"Alpha values : {len(alpha_values)}")
    print(f"Beta values  : {len(beta_values)}")
    print(f"Gamma values : {len(gamma_values)}")


if __name__ == "__main__":
    main()
