"""Generate legacy one-at-a-time parameter samples."""

import json
import os

import numpy as np
import pandas as pd


BASELINE_PARAMS = {
    "sigma_signal": 0.06,
    "sigma_belief": 0.025,
    "rho_y": 0.985,
    "sigma_y": 0.025,
    "alpha_social": 0.75,
    "beta": 1.0,
    "gamma": 0.90,
    "trade_sensitivity": 2.4,
    "price_impact": 0.02,
    "sigma_price": 0.10,
}


PARAM_SPECS = {
    "sigma_signal": {
        "method": "log_uniform",
        "low": 1e-3,
        "high": 3e-1,
        "n_samples": 200,
    },
    "sigma_belief": {
        "method": "log_uniform",
        "low": 5e-4,
        "high": 2e-1,
        "n_samples": 200,
    },
    "rho_y": {
        "method": "one_minus_log_uniform",
        "low": 0.50,
        "high": 0.9999,
        "n_samples": 200,
    },
    "sigma_y": {
        "method": "log_uniform",
        "low": 5e-4,
        "high": 2e-1,
        "n_samples": 200,
    },
    "alpha_social": {
        "method": "uniform",
        "low": 0.0,
        "high": 0.99,
        "n_samples": 1000,
    },
    "beta": {
        "method": "log_uniform",
        "low": 1e-2,
        "high": 1e3,
        "n_samples": 1000,
    },
    "gamma": {
        "method": "uniform",
        "low": 0.0,
        "high": 0.999,
        "n_samples": 200,
    },
    "trade_sensitivity": {
        "method": "log_uniform",
        "low": 5e-2,
        "high": 2e1,
        "n_samples": 1000,
    },
    "price_impact": {
        "method": "log_uniform",
        "low": 1e-4,
        "high": 5e-1,
        "n_samples": 1000,
    },
    "sigma_price": {
        "method": "log_uniform",
        "low": 1e-3,
        "high": 5e-1,
        "n_samples": 200,
    },
}


def sample_uniform(rng, low, high, size):
    return rng.uniform(low, high, size=size)


def sample_log_uniform(rng, low, high, size):
    return np.exp(
        rng.uniform(
            np.log(low),
            np.log(high),
            size=size,
        )
    )


def sample_one_minus_log_uniform(
    rng,
    low_rho,
    high_rho,
    size,
):
    d_low = 1.0 - high_rho
    d_high = 1.0 - low_rho
    d = sample_log_uniform(
        rng,
        d_low,
        d_high,
        size,
    )
    return 1.0 - d


def draw_values(rng, spec):
    method = spec["method"]

    if method == "uniform":
        return sample_uniform(
            rng,
            spec["low"],
            spec["high"],
            spec["n_samples"],
        )

    if method == "log_uniform":
        return sample_log_uniform(
            rng,
            spec["low"],
            spec["high"],
            spec["n_samples"],
        )

    if method == "one_minus_log_uniform":
        return sample_one_minus_log_uniform(
            rng,
            spec["low"],
            spec["high"],
            spec["n_samples"],
        )

    raise ValueError(f"Unknown method: {method}")


def generate_oat_parameter_samples(
    outdir="oat_parameter_samples",
    seed=2026,
):
    """Generate the original legacy OAT parameter samples."""
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(seed)

    for param_name, spec in PARAM_SPECS.items():
        values = draw_values(rng, spec)

        rows = []

        for i, val in enumerate(values):
            row = {
                "sample_id": i,
                "varied_parameter": param_name,
            }

            for param, base_val in BASELINE_PARAMS.items():
                row[param] = base_val

            row[param_name] = float(val)
            rows.append(row)

        df = pd.DataFrame(rows)

        csv_path = os.path.join(
            outdir,
            f"{param_name}_samples.csv",
        )
        df.to_csv(csv_path, index=False)

        print(
            f"Saved {csv_path} with {len(df)} samples"
        )

    with open(
        os.path.join(outdir, "baseline_params.json"),
        "w",
    ) as file:
        json.dump(BASELINE_PARAMS, file, indent=2)

    with open(
        os.path.join(outdir, "parameter_specs.json"),
        "w",
    ) as file:
        json.dump(PARAM_SPECS, file, indent=2)
