"""Detect network importance in legacy OAT results."""

import os

import pandas as pd


THRESHOLDS = {
    "explosion_rate_relative_gap": 0.02,
    "mean_abs_return_relative_gap": 0.10,
    "mean_belief_var_relative_gap": 0.10,
    "mean_avg_abs_position_relative_gap": 0.005,
}


def classify_row(row):
    """Apply the legacy relative-gap network-importance criteria."""
    flags = {}

    for key, threshold in THRESHOLDS.items():
        val = row.get(key, None)
        flags[f"flag_{key}"] = int(
            pd.notna(val) and val >= threshold
        )

    overall = int(any(flags.values()))
    flags["network_matters"] = overall

    # Preserve the legacy scoring behavior during restructuring.
    flags["network_importance_score"] = sum(flags.values())

    return flags


def detect_network_importance(
    infile=(
        "oat_topology_comparison/"
        "topology_gap_by_parameter_value.csv"
    ),
    outdir="oat_network_importance",
):
    """Classify OAT parameter values by legacy network-importance rules."""
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(infile)

    flag_rows = []

    for _, row in df.iterrows():
        base = row.to_dict()
        flags = classify_row(base)
        base.update(flags)
        flag_rows.append(base)

    out_df = pd.DataFrame(flag_rows)

    out_csv = os.path.join(
        outdir,
        "network_importance_detection.csv",
    )
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    flagged = out_df[
        out_df["network_matters"] == 1
    ].copy()

    flagged_csv = os.path.join(
        outdir,
        "network_importance_flagged_only.csv",
    )
    flagged.to_csv(flagged_csv, index=False)
    print(f"Saved: {flagged_csv}")

    summary = (
        out_df.groupby(
            "varied_parameter",
            as_index=False,
        )
        .agg(
            n_values=("varied_value", "count"),
            n_flagged=("network_matters", "sum"),
            share_flagged=("network_matters", "mean"),
            avg_importance_score=(
                "network_importance_score",
                "mean",
            ),
            max_importance_score=(
                "network_importance_score",
                "max",
            ),
        )
        .sort_values(
            ["share_flagged", "avg_importance_score"],
            ascending=False,
        )
    )

    summary_csv = os.path.join(
        outdir,
        "network_importance_summary_by_parameter.csv",
    )
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    return out_df, flagged, summary
