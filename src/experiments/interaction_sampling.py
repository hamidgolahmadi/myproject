"""Parameter-grid generation for legacy ABG interaction experiments."""

import os

import numpy as np
import pandas as pd


BASELINE_PARAMS = {
    "sigma_signal": 0.06,
    "sigma_belief": 0.025,
    "rho_y": 0.985,
    "sigma_y": 0.025,
    "trade_sensitivity": 2.4,
    "price_impact": 0.02,
    "sigma_price": 0.10,
}


FULL_ALPHA_VALUES = np.array(
    [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00],
    dtype=float,
)
FULL_BETA_VALUES = np.geomspace(0.5, 50.0, 25)
FULL_GAMMA_VALUES = np.geomspace(0.01, 1.0, 25)


COARSE_ALPHA_VALUES = np.array(
    [0.125, 0.15, 0.175, 0.20, 0.225, 0.25],
    dtype=float,
)
COARSE_BETA_VALUES = np.geomspace(0.5, 50.0, 10)
COARSE_GAMMA_VALUES = np.geomspace(0.01, 1.0, 10)


def build_abg_grid(alpha_values, beta_values, gamma_values):
    """Build an alpha-beta-gamma grid using the legacy ordering."""
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
                    **BASELINE_PARAMS,
                }
                rows.append(row)
                sample_id += 1

    return pd.DataFrame(rows)


def save_abg_grid(
    alpha_values,
    beta_values,
    gamma_values,
    outpath,
):
    """Build and save one legacy ABG interaction parameter grid."""
    outdir = os.path.dirname(outpath)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    df = build_abg_grid(
        alpha_values,
        beta_values,
        gamma_values,
    )
    df.to_csv(outpath, index=False)

    return df


def generate_full_abg_grid(
    outpath="interaction_parameter_samples/alpha_beta_gamma_grid.csv",
):
    """Generate the original full ABG interaction grid."""
    df = save_abg_grid(
        FULL_ALPHA_VALUES,
        FULL_BETA_VALUES,
        FULL_GAMMA_VALUES,
        outpath,
    )

    print(f"Saved {outpath}")
    print(f"Total samples: {len(df)}")
    print(f"Alpha values : {len(FULL_ALPHA_VALUES)}")
    print(f"Beta values  : {len(FULL_BETA_VALUES)}")
    print(f"Gamma values : {len(FULL_GAMMA_VALUES)}")

    return df


def generate_coarse_abg_grid(
    outpath="interaction_parameter_samples/abg_alpha125_250_coarse.csv",
):
    """Generate the original refined coarse ABG grid."""
    df = save_abg_grid(
        COARSE_ALPHA_VALUES,
        COARSE_BETA_VALUES,
        COARSE_GAMMA_VALUES,
        outpath,
    )

    print(f"Saved {outpath}")
    print(f"Total samples: {len(df)}")
    print(
        f"Alpha values : {len(COARSE_ALPHA_VALUES)} "
        f"-> {COARSE_ALPHA_VALUES.tolist()}"
    )
    print(
        f"Beta values  : {len(COARSE_BETA_VALUES)} "
        f"-> min={COARSE_BETA_VALUES.min():.6g}, "
        f"max={COARSE_BETA_VALUES.max():.6g}"
    )
    print(
        f"Gamma values : {len(COARSE_GAMMA_VALUES)} "
        f"-> min={COARSE_GAMMA_VALUES.min():.6g}, "
        f"max={COARSE_GAMMA_VALUES.max():.6g}"
    )

    return df
