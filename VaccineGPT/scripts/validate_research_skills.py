from __future__ import annotations

import re
from pathlib import Path


REQUIRED = {
    "literature_retrieval.md": ["OpenAlex", "PubMed", "deduplic", "checksum"],
    "literature_review.md": ["evidence", "contradict", "citation", "leakage"],
    "nsfc_project.md": ["hypothesis", "controls", "Budget", "risk"],
    "report_generation.md": ["Markdown", "PPT", "SVG", "QA"],
    "research_pipeline.md": ["manifest", "resumable", "citation"],
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "skills"
    failures = []
    for filename, terms in REQUIRED.items():
        text = (root / filename).read_text(encoding="utf-8")
        missing = [term for term in terms if not re.search(term, text, re.I)]
        if missing:
            failures.append({"file": filename, "missing": missing})
    if failures:
        raise SystemExit(f"skill validation failed: {failures}")
    print(f"validated {len(REQUIRED)} integrated research skills")


if __name__ == "__main__":
    main()
