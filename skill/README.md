# VaccineGPT integrated research skills

This directory distills the local research skills into reusable, testable
workflows. It is intentionally at the repository root, beside `reference/`,
so research workflows remain reusable without being mixed with model code.

## Included skills

| Skill | Purpose | Primary outputs |
|---|---|---|
| `literature_retrieval.md` | Query expansion, OpenAlex/PubMed/Europe PMC retrieval, OA download, DOI deduplication and provenance | `references.jsonl`, `abstracts.txt`, retrieval audit |
| `literature_review.md` | PRISMA-style screening, evidence extraction, contradiction analysis, synthesis and citation quality checks | Markdown review, evidence table, quality report |
| `nsfc_project.md` | Scientific question, innovation, aims, work packages, risk controls, budget and feasibility | NSFC-style Markdown proposal |
| `report_generation.md` | Academic Markdown, PPT, tables, diagrams, Mermaid and publication-ready figures | `.md`, `.pptx`, `.svg/.png`, QA report |
| `research_pipeline.md` | Orchestrates retrieval → review → project design → report QA | run manifest and continuation record |

## Safety and quality gates

1. Never treat an HTML error page as a biological dataset.
2. Record URL, release, retrieval time, checksum, license and parser version.
3. Separate discovery metadata from validated evidence.
4. Do not invent sample sizes, p-values, effect sizes, or citations.
5. Keep a rejected-record file and an explicit uncertainty field.
6. Run `python scripts/validate_research_skills.py` after modifying these
   skill documents.
