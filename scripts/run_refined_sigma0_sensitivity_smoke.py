"""Run the pre-freeze common-random-number sensitivity smoke for sigma_0."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.refined import (
    Sigma0SensitivityProtocol,
    run_sigma0_sensitivity_smoke,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "refined" / "sigma0_sensitivity_smoke"


def _write_outputs(result) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "sigma0_sensitivity_summary.csv"
    metadata_path = OUTPUT_DIR / "sigma0_sensitivity_metadata.json"

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sigma_0",
                "metric",
                "count",
                "mean",
                "median",
                "minimum",
                "maximum",
            ),
        )
        writer.writeheader()
        for row in result.pooled_rows:
            writer.writerow(asdict(row))

    metadata = {
        "purpose": (
            "pre-freeze OAT sensitivity for sigma_0; common graph/shock/initial-state "
            "randomness; pooled absolute diagnostics only; not topology ranking"
        ),
        "experiment_seed": result.protocol.experiment_seed,
        "n_replications": result.protocol.n_replications,
        "sigma0_values": list(result.protocol.sigma0_values),
        "n_agents": result.candidate.n_agents,
        "horizon": result.candidate.horizon,
        "topology_labels": [
            specification.topology_label
            for specification in result.candidate.topology_specifications
        ],
        "base_parameters": asdict(result.candidate.parameters),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary_path, metadata_path


def _median(result, sigma_0: float, metric: str) -> float:
    for row in result.rows_for(sigma_0):
        if row.metric == metric:
            return row.median
    raise KeyError(metric)


def main() -> None:
    protocol = Sigma0SensitivityProtocol()
    result = run_sigma0_sensitivity_smoke(protocol=protocol)
    summary_path, metadata_path = _write_outputs(result)

    print("Refined sigma_0 sensitivity smoke completed")
    print(
        f"common paired replications={protocol.n_replications}, "
        f"sigma_0 values={len(protocol.sigma0_values)}"
    )
    print("Purpose: pre-freeze absolute-scale sensitivity only; do not rank topologies.")
    print()
    print(
        "sigma_0      rep/sigma0   mean_W_mob   max_W_mob    final_W_dist  "
        "ret_std      rms_mispricing  projection"
    )
    for sigma_0 in protocol.sigma0_values:
        print(
            f"{sigma_0:<12.6g} "
            f"{_median(result, sigma_0, 'median_reputation_scale_to_sigma0'):>11.4f} "
            f"{_median(result, sigma_0, 'mean_attention_mobility'):>12.6f} "
            f"{_median(result, sigma_0, 'max_attention_mobility'):>12.6f} "
            f"{_median(result, sigma_0, 'final_attention_distance_from_initial'):>13.6f} "
            f"{_median(result, sigma_0, 'return_std'):>11.6f} "
            f"{_median(result, sigma_0, 'rms_mispricing'):>15.6f} "
            f"{_median(result, sigma_0, 'execution_projection_fraction'):>11.6f}"
        )

    print()
    print(f"summary:  {summary_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
