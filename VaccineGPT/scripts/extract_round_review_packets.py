from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def normalize(value: str) -> str:
    return " ".join((value or "").split())


def section_text(root: ElementTree.Element) -> list[dict[str, str]]:
    sections = []
    for sec in root.findall(".//sec"):
        title_node = sec.find("title")
        title = normalize("".join(title_node.itertext())) if title_node is not None else "untitled section"
        text = normalize(" ".join("".join(node.itertext()) for node in sec.findall(".//p")))
        if text:
            sections.append({"title": title, "text": text})
    return sections


def extract_packet(ref: dict[str, Any], maximum_per_section: int = 1800) -> dict[str, Any]:
    path = Path(ref["fulltext_path"])
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    with gzip.open(path, "rb") as stream:
        xml_bytes = stream.read()
    root = ElementTree.fromstring(xml_bytes)
    sections = section_text(root)
    selected = []
    for section in sections:
        heading = section["title"].casefold()
        if any(token in heading for token in ("method", "result", "discussion", "conclusion", "data", "limitation")):
            selected.append(
                {
                    "title": section["title"],
                    "text": section["text"][:maximum_per_section],
                    "truncated": len(section["text"]) > maximum_per_section,
                }
            )
    return {
        "round": ref["round"],
        "pmcid": ref.get("pmcid"),
        "pmid": ref.get("pmid"),
        "doi": ref.get("doi"),
        "title": ref.get("title"),
        "authors": ref.get("authorString"),
        "year": ref.get("pubYear"),
        "license": ref.get("license"),
        "retrieved_at": ref.get("retrieved_at_utc"),
        "fulltext_xml_sha256": ref.get("fulltext_xml_sha256"),
        "fulltext_bytes": len(xml_bytes),
        "abstract": normalize(ref.get("abstractText") or ""),
        "methods_results_discussion_sections": selected,
        "review_scope": "selected passages from downloaded open-access XML; reviewer must inspect context before claiming a result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic full-text review packets by GCDTO round.")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=261007)
    args = parser.parse_args()
    source = args.campaign_root / "rounds" / f"round_{args.round:02d}_references.jsonl"
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < args.sample_size:
        raise ValueError(
            f"round {args.round} has {len(rows)} references; need {args.sample_size} for review packet"
        )
    rng = random.Random(args.seed + args.round)
    selected = sorted(rng.sample(rows, args.sample_size), key=lambda row: row["pmcid"])
    packets = [extract_packet(row) for row in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(packet, ensure_ascii=False) + "\n" for packet in packets),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "round": args.round,
                "sample_size": len(packets),
                "seed": args.seed + args.round,
                "output": str(args.output),
                "manual_review_required": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
