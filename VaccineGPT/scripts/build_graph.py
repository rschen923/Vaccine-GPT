from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts import validate_graph


RELATIONS = {
    "operon": 0,
    "coexpression": 1,
    "coexpr": 1,
    "coadaptation": 2,
    "interaction": 3,
    "ppi": 3,
    "clef_eei": 3,
    "dna_regulation": 4,
    "dna_element_proximity": 5,
}
V1_RELATIONS = {"operon", "coexpression", "coexpr", "coadaptation", "interaction", "ppi", "clef_eei"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the six-slot GCE graph while retaining edge evidence and split boundaries."
    )
    parser.add_argument("--nodes", required=True, help="JSON/JSONL node records with gene_id")
    parser.add_argument("--edges", nargs="+", required=True, help="TSV files: src,dst,relation[,confidence]")
    parser.add_argument("--splits", required=True, help="split_manifest.json")
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--track", choices=("v1_protein", "v2_dna"), default="v1_protein")
    args = parser.parse_args()
    nodes_path = Path(args.nodes)
    if nodes_path.suffix == ".json":
        nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
        node_ids = [str(item["gene_id"] if isinstance(item, dict) else item) for item in nodes]
    else:
        node_ids = [
            json.loads(line)["gene_id"]
            for line in nodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("graph node IDs must be unique")
    node_index = {gene: index for index, gene in enumerate(node_ids)}
    split_manifest = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    split_by_gene = {gene: split for split, genes in split_manifest.items() for gene in genes}
    src: list[int] = []
    dst: list[int] = []
    edge_type: list[int] = []
    edge_weight: list[float] = []
    evidence: list[str] = []
    edge_sources: list[str] = []
    counts: Counter[str] = Counter()
    for path in args.edges:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"src", "dst", "relation"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path}: edge table requires columns {sorted(required)}")
            for row in reader:
                left, right = str(row["src"]), str(row["dst"])
                raw_relation = str(row["relation"]).strip().lower()
                relation = raw_relation
                if relation not in RELATIONS:
                    counts["unknown_relation"] += 1
                    continue
                is_dna_relation = relation in {"dna_regulation", "dna_element_proximity"}
                if is_dna_relation and args.track != "v2_dna":
                    raise ValueError(f"{relation} is only permitted in the v2 DNA graph")
                if args.track == "v1_protein" and relation not in V1_RELATIONS:
                    raise ValueError(f"{relation} is not part of the v1 protein graph")
                if left not in node_index or right not in node_index or left == right:
                    counts["unknown_or_self_node_edge"] += 1
                    continue
                if left not in split_by_gene or right not in split_by_gene:
                    counts["missing_split_assignment_edges_removed"] += 1
                    continue
                if split_by_gene[left] != split_by_gene[right]:
                    counts["cross_split_edges_removed"] += 1
                    continue
                try:
                    confidence = float(row.get("confidence", 1.0) or 1.0)
                except ValueError as error:
                    raise ValueError(f"{path}: invalid confidence for {left}->{right}") from error
                canonical_relation = "coexpression" if relation == "coexpr" else (
                    "interaction" if relation in {"ppi", "clef_eei"} else relation
                )
                if canonical_relation == "coexpression":
                    if not -1 <= confidence <= 1:
                        raise ValueError(f"{path}: Pearson correlation must be in [-1, 1]")
                    edge_value = max(0.0, confidence**2)
                    threshold_value = abs(confidence)
                else:
                    if not 0 <= confidence <= 1:
                        raise ValueError(f"{path}: edge confidence must be in [0, 1]")
                    edge_value = confidence
                    threshold_value = confidence
                threshold = {
                    "operon": 0.0,
                    "coexpression": 0.8,
                    "coadaptation": 0.9,
                    "interaction": 0.5 if relation == "clef_eei" else 0.7,
                    "dna_regulation": 0.0,
                    "dna_element_proximity": 0.0,
                }[canonical_relation]
                if threshold_value < threshold:
                    counts[f"{canonical_relation}_below_threshold"] += 1
                    continue
                edge_level = str(row.get("evidence_level") or ("E3" if canonical_relation in {"interaction", "dna_regulation", "dna_element_proximity"} else "E2"))
                if edge_level not in {"E1", "E2", "E3", "E4"}:
                    raise ValueError(f"{path}: unsupported evidence_level {edge_level}")
                source = str(row.get("source") or Path(path).name)
                for a, b in ((left, right), (right, left)):
                    src.append(node_index[a])
                    dst.append(node_index[b])
                    edge_type.append(RELATIONS[canonical_relation])
                    edge_weight.append(edge_value)
                    evidence.append(edge_level)
                    edge_sources.append(source)
                counts[canonical_relation] += 2
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    lineage["batch_id"] = args.batch_id
    record = {
        "node_ids": node_ids,
        "edge_index": [src, dst],
        "edge_type": edge_type,
        "edge_weight": edge_weight,
        "edge_evidence": evidence,
        "edge_sources": edge_sources,
        "num_rels": 6,
        "track": args.track,
        "lineage": lineage,
    }
    validate_graph(record)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nodes": len(node_ids),
                "directed_edges": len(edge_type),
                "relation_counts": dict(counts),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
