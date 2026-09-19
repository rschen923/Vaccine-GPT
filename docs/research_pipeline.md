# Literature evidence pipeline

`VaccineGPT/scripts/research/literature_pipeline.py` is an evidence-first,
resumable collector. It uses only `urllib` and `json` for network and
serialization; no paper is invented when a provider is unavailable.

## Run

```bash
python VaccineGPT/scripts/research/literature_pipeline.py self-test
python VaccineGPT/scripts/research/literature_pipeline.py retrieve \
  VaccineGPT/configs/literature/query_manifest.json literature-output
python VaccineGPT/scripts/research/literature_pipeline.py qa \
  VaccineGPT/configs/literature/query_manifest.json literature-output
```

`retrieve` writes `papers.jsonl`, `failures.jsonl`, `completed_queries.json`,
and a summary. Query completion is persisted after each query, so rerunning
resumes safely. Network failures remain explicit failure records. Papers are
restricted to 2024-09-19 through 2026-09-19 (inclusive), and are deduplicated
by DOI, PMID, PMCID, or normalized title. `resolve_fulltext` (library API)
downloads OA content and records SHA-256 checksums; mismatches are rejected.

Screening must use `screen(paper, decision, reason)`, where every decision has
an explicit reason. Evidence and claim consumers should append JSONL records
with `record_type` values `evidence` and `claim`, retaining the paper key and
provenance. `citation_gbt` exports a GB/T 7714-2025-ish citation while source
metadata remains in the paper's provenance fields.

QA is deterministic: it checks dates, duplicate identifiers, provenance, and
per-section minimum plus review/research composition quotas. An empty or
partially retrieved manifest therefore fails honestly; the pipeline never
claims a target number of papers without records.
