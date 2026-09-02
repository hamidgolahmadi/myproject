"""Run the provisional refined-baseline scale/non-degeneracy smoke job."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.refined.baseline_specification import (  # noqa: E402
    first_refined_baseline_candidate,
)
from src.experiments.refined.market_smoke import (  # noqa: E402
    MarketScaleSmokeProtocol,
    run_first_refined_baseline_scale_smoke,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the small paired refined-baseline scale smoke. "
            "This is a pre-freeze diagnostic, not a topology comparison."
        )
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Number of paired smoke replications (default: 5).",
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=2026090203,
        help="Dedicated smoke seed namespace, disjoint from D042 calibration seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "refined" / "baseline_scale_smoke",
    )
    return parser.parse_args()


def write_result(result, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "scale_smoke_records.csv"
    summary_path = output_dir / "scale_smoke_pooled_summary.csv"
    metadata_path = output_dir / "scale_smoke_metadata.json"

    record_rows = [asdict(record) for record in result.records]
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record_rows[0]))
        writer.writeheader()
        writer.writerows(record_rows)

    summary_rows = [asdict(summary) for summary in result.pooled_summary]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    candidate = result.candidate
    metadata = {
        "purpose": "pre-freeze absolute scale/non-degeneracy smoke; not topology ranking",
        "experiment_seed": result.protocol.experiment_seed,
        "n_replications": result.protocol.n_replications,
        "n_runs": len(result.records),
        "n_agents": candidate.n_agents,
        "horizon": candidate.horizon,
        "k": candidate.k,
        "p_sw": candidate.p_sw,
        "a0": candidate.a0,
        "action_saturation_cutoff": 0.99,
        "parameters": asdict(candidate.parameters),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return records_path, summary_path, metadata_path


def main() -> None:
    args = parse_arguments()
    protocol = MarketScaleSmokeProtocol(
        experiment_seed=args.experiment_seed,
        n_replications=args.replications,
    )
    candidate = first_refined_baseline_candidate()
    result = run_first_refined_baseline_scale_smoke(
        protocol=protocol,
        candidate=candidate,
    )
    records_path, summary_path, metadata_path = write_result(result, args.output_dir)

    print("Refined baseline scale smoke completed")
    print(
        f"paired replications={protocol.n_replications}, "
        f"runs={len(result.records)}, N={candidate.n_agents}, T={candidate.horizon}"
    )
    print("Purpose: absolute scale/non-degeneracy only; do not rank topologies from this smoke.")
    print()
    print(f"{'metric':42s} {'median':>12s} {'min':>12s} {'max':>12s}")
    for summary in result.pooled_summary:
        print(
            f"{summary.metric:42s} "
            f"{summary.median:12.6g} {summary.minimum:12.6g} {summary.maximum:12.6g}"
        )
    print()
    print(f"raw records: {records_path}")
    print(f"pooled summary: {summary_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
