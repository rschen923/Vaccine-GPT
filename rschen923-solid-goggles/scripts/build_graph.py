from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contracts import validate_graph


RELATIONS = {
    "ppi": 0, "operon": 1, "coexpr": 2, "homology": 3,
    "neighbor": 4, "regulation": 5, "pathway": 6,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UDC-04 seven-relation graph from edge tables")
    parser.add_argument("--nodes", required=True, help="JSON list or JSONL node records with gene_id")
    parser.add_argument("--edges", nargs="+", required=True, help="TSV files: src,dst,relation[,confidence]")
    parser.add_argument("--splits", required=True, help="split_manifest.json")
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    nodes_path = Path(args.nodes)
    if nodes_path.suffix == ".json":
        nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
        node_ids = [str(item["gene_id"] if isinstance(item, dict) else item) for item in nodes]
    else:
        node_ids = [
            json.loads(line)["gene_id"] for line in nodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    node_index = {gene: index for index, gene in enumerate(node_ids)}
    split_manifest = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    split_by_gene = {
        gene: split for split, genes in split_manifest.items() for gene in genes
    }
    src, dst, edge_type = [], [], []
    for path in args.edges:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                left, right = str(row["src"]), str(row["dst"])
                relation = str(row["relation"]).lower()
                if left not in node_index or right not in node_index or relation not in RELATIONS:
                    continue
                if split_by_gene.get(left) != split_by_gene.get(right):
                    continue
                confidence = float(row.get("confidence", 1.0) or 1.0)
                if relation in ("ppi", "coexpr") and confidence < 0.7:
                    continue
                src.extend((node_index[left], node_index[right]))
                dst.extend((node_index[right], node_index[left]))
                edge_type.extend((RELATIONS[relation], RELATIONS[relation]))
    lineage = json.loads(Path(args.lineage).read_text(encoding="utf-8"))
    record = {
        "node_ids": node_ids, "edge_index": [src, dst], "edge_type": edge_type,
        "num_rels": 7, "track": "SOM", "lineage": lineage,
    }
    validate_graph(record)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record), encoding="utf-8")
    print(json.dumps({"nodes": len(node_ids), "directed_edges": len(edge_type), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
