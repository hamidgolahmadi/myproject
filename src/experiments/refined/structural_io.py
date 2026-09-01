"""CSV/JSON output helpers for refined structural-validation ensembles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .structural import StructuralEnsembleResult


_METRICS = (
    "in_degree_gini",
    "hub_link_share",
    "global_clustering",
    "average_path_length_lcc",
    "largest_component_share",
)


def write_structural_result(
    result: StructuralEnsembleResult,
    *,
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Write raw graph-level records, descriptive summaries, and metadata.

    The raw CSV is the primary analysis object.  The summary CSV is descriptive
    only; no graph-level observations are discarded from the returned result.
    """

    if not isinstance(result, StructuralEnsembleResult):
        raise TypeError("result must be a StructuralEnsembleResult")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_path = output_path / "structural_graph_records.csv"
    summary_path = output_path / "structural_summary.csv"
    metadata_path = output_path / "structural_metadata.json"

    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "replication_id",
                "topology_label",
                "graph_seed",
                "n_agents",
                "total_links",
                "mean_out_degree",
                "in_degree_gini",
                "hub_q",
                "hub_link_share",
                "global_clustering",
                "average_path_length_lcc",
                "largest_component_share",
            ],
        )
        writer.writeheader()
        for record in result.records:
            diagnostics = record.diagnostics
            writer.writerow(
                {
                    "replication_id": record.replication_id,
                    "topology_label": record.topology_label,
                    "graph_seed": record.graph_seed,
                    "n_agents": diagnostics.n_agents,
                    "total_links": diagnostics.total_links,
                    "mean_out_degree": diagnostics.mean_out_degree,
                    "in_degree_gini": diagnostics.in_degree_gini,
                    "hub_q": diagnostics.hub_q,
                    "hub_link_share": diagnostics.hub_link_share,
                    "global_clustering": diagnostics.global_clustering,
                    "average_path_length_lcc": diagnostics.average_path_length_lcc,
                    "largest_component_share": diagnostics.largest_component_share,
                }
            )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "topology_label",
                "metric",
                "count",
                "mean",
                "std",
                "minimum",
                "q25",
                "median",
                "q75",
                "maximum",
            ],
        )
        writer.writeheader()
        for topology_label in result.topology_labels:
            summary = result.summary_for(topology_label)
            for metric in _METRICS:
                values = getattr(summary, metric)
                writer.writerow(
                    {
                        "topology_label": topology_label,
                        "metric": metric,
                        "count": values.count,
                        "mean": values.mean,
                        "std": values.std,
                        "minimum": values.minimum,
                        "q25": values.q25,
                        "median": values.median,
                        "q75": values.q75,
                        "maximum": values.maximum,
                    }
                )

    metadata = {
        "experiment_seed": result.experiment_seed,
        "n_agents": result.n_agents,
        "n_replications": result.n_replications,
        "hub_q": result.q,
        "topologies": [
            {
                "topology_label": spec.topology_label,
                "kind": spec.kind,
                "k": spec.k,
                "p_sw": spec.p_sw,
                "a0": spec.a0,
            }
            for spec in result.specifications
        ],
        "n_records": len(result.records),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return raw_path, summary_path, metadata_path
