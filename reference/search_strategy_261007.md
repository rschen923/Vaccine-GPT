# Search strategy and quota audit — 2026-10-07

## Question

For a staged bacterial vaccine-target prioritization system, what publicly
validated methods and code support (1) protein/DNA sequence representation,
(2) bacterial gene/functional annotation, (3) experiment-aware labels,
(4) graph/context modeling, (5) uncertainty and multi-objective ranking, and
(6) leakage-safe evaluation?

## Planned eligibility

- Primary research, benchmark, methods, and official software documentation;
- public full text required for any item counted toward the user's quota;
- protein/DNA model claims must match the reported task/modality;
- bacterial gene-function and vaccine claims must retain organism, assay,
  strain, and experimental endpoint;
- exclude abstract-only records from “full-text reviewed” counts;
- distinguish a published article from its preprint rather than counting both
  as independent evidence;
- separate repository license from weight, dataset, and dependency terms.

## Sources and initial-pass retrieval status

| Source | Actual use in this pass | Limitation |
|---|---|---|
| Europe PMC | Broad discovery; reported 434 title/abstract-query hits | Search was broad and not screened to eligible full-text records; exact per-request query URL/response not retained |
| PubMed | Broad discovery; reported 3,039 results | Search hits are not papers reviewed or quota counts; exact request URL/response not retained |
| Crossref | DOI metadata checking | Raw response snapshots and field conflict log not retained |
| PMC, arXiv, publisher records | Primary full-text/metadata routes for selected records | Relevant portions of six full texts inspected; not a 2,000-work full-text screen |
| GitHub primary repositories | Selected README/license/source paths across 22 repositories | No systematic maintenance/archive audit or dependency/weight license review for every repository |

The source pass did not preserve a complete append-only query manifest with
exact raw request URLs, UTC timestamps, unmodified responses, and response
hashes. Accordingly, it is **not** a compliant persistent evidence-pipeline
run. The current partial evidence ledger uses only date-level retrieval status
and marks this limitation.

## Initial reconnaissance coverage (before the full campaign)

| Measure | Requested | Actually reviewed/listed | Status |
|---|---:|---:|---|
| GCDTO rounds | 10 completed rounds | 0 completed quota-compliant rounds | Not met |
| Full-text papers | ≥200 per round (≥2,000 round-references) | 6 full texts had relevant portions opened | Not met |
| Public code repositories | ≥20 per round (≥200 round-references) | 22 repositories listed with selected paths | Not met |
| Full-text access verification | Every counted paper | Verified for the six opened works; Evo 2 only front matter | Partial |
| Independent index coverage | ≥2 scholarly indexes | PubMed + Europe PMC; Crossref for DOI checks | Partial |
| Raw responses/query manifest | Exact request + raw response | Not preserved by the delegated search pass | Not met |

The requested 200 full-text papers per round cannot be represented by broad
search hit counts. Repeating a citation across rounds without each round's
actual argument and evidence review is not a valid substitute for the quota.

## Planned stop rule for a compliant continuation

For each round: freeze the round's question, inclusion criteria and query set;
retrieve ≥200 deduplicated OA/repository full texts and ≥20 primary-source code
repositories; screen paper versions and licenses; inspect relevant full-text
sections and repository paths; run duplicate/identifier/retraction checks;
save raw snapshots and a schema-validated ledger before writing the round's
正/反/合 and falsifiable detector. Stop or label a round `blocked` when an
item has only abstract/landing-page access or provenance is unavailable.

## 2026-10-07 campaign execution addendum

A second, auditable retrieval pass subsequently queried Europe PMC using the
ten query strings in `VaccineGPT/scripts/run_literature_campaign.py`, with
`OPEN_ACCESS:Y`, title/abstract filters, `resultType=core`, and up to 1,000
search records per round. Exact request URLs and raw JSON response snapshots
plus SHA-256 hashes are in
`..\data\research_campaign_261007\query_manifest.jsonl` and
`raw/round_XX_search_response.json`. Each selected article has a fetched
Europe PMC `fullTextXML` file, source URL, byte count, and SHA-256 recorded in
the round reference file and normalized ledger.

To satisfy the raw-before-parse provenance ordering, the ten exact query
requests were re-snapshotted in
`..\data\research_campaign_261007\strict_raw_queries\` by
`VaccineGPT/scripts/snapshot_europepmc_queries.py`; that script writes raw
response bytes before `json.loads`, and `strict_query_manifest.jsonl` records
request URL, receive/persist/parse timestamps, raw SHA-256, and confirmation
that all 200 PMCID references in each existing round were present.

**Retrieval quota result:** all 10 themed rounds have 200/200 unique-PMCID
references with downloadable XML; total round-reference count 2,000 and
deduplicated unique-work count 1,806 (identifier priority DOI > PMID > PMCID >
normalized title). The audit verifies 2,000/2,000 cached XML byte hashes, with
zero full-text integrity errors and zero DOI/PMID/PMCID title conflicts.
There are 100 deterministic sample rows (10 per round) in
`manual_verification_sample.jsonl`; 90 round-reference rows (89 unique PMCID
works) are marked as AI-subagent reviewed on selected excerpts; 10 for round 5
are marked blocked by the biological-risk filter. Four previously reviewed
campaign papers not in the random sample plus two separately read arXiv
sources raise the unique sampled-section review count to 95 works (93 in the
campaign; 1,713 campaign works have not had individual claim review).

**Important distinction:** the 200-per-round criterion is met as a
**full-text-retrieval/reference availability quota**. It is **not** a claim
that 2,000 full papers were individually read or that all extracted claims
were manually verified. AI-subagent packet reviews cover 10 randomly selected
articles each for rounds 1–4 and 6–10; round 5’s packet review was blocked by
the research agent’s biological-risk filter and was not retried through an
alternate route. Substantive review of the remaining 1,713 campaign works is pending.
The round-specific GCDTO dossiers state this boundary and provide 200-source
indexes.

## Actual versus requested coverage (after campaign retrieval)

| Measure | Requested | Verified | Status |
|---|---:|---:|---|
| Themed research rounds | 10 | 10 query snapshots | retrieval stage complete |
| OA fullTextXML references | ≥200 each round | 200 in every round; 2,000 round references | retrieval quota met |
| Deduplicated full-text works | unprescribed | 1,806 | identifier-deduplicated ledger |
| Full-text byte/hash integrity | every counted full text | 2,000/2,000 XML hashes verified | passed |
| AI-assisted full-text excerpt review | evidence-grounded review | 90 round-reference rows (89 unique campaign works); 10 for round 5 blocked | partial |
| Public code repositories | ≥20 each round | 202 round-repository references; 192 global distinct repositories | retrieval/inspection quota met |
| Per-claim/quantitative extraction | every substantive claim | abstracts and exact section indexes for corpus; only sampled excerpt review | incomplete |

The reports must not equate “downloaded open full text” with “manually
validated every substantive paper claim”.

### PRISMA-like retrieval flow

Counts below are the ten Europe PMC query totals; round-level query results can
overlap and therefore the starting count is not a unique-paper denominator.
Only the first search page was screened per query (maximum 1,000 results; round
5 had 922):

```text
10 themed OA queries
  → 73,212 reported search hits (overlapping; not unique)
  → 9,922 first-page records screened by title/abstract + OA + PMCID
  → 9,449 topic/OA candidates
  → 2,000 fullTextXML items fetched and hash-verified (200 per round)
  → 1,806 works after DOI > PMID > PMCID > normalized-title deduplication
  → 100 random full-text packets sampled
      → 90 sampled round-reference excerpts AI-subagent reviewed (89 unique works)
      → 6 earlier source works also summarized (4 campaign overlaps + 2 non-campaign arXiv)
      → 10 round-5 packet items blocked by safety review; not rerouted
  → 93 campaign works + 2 non-campaign arXiv works with selected-section review
  → 1,713 campaign works without individual claim review
```

The 7,449 screened OA/topic candidates not fetched as complete XML were outside
the per-round target after rank-order selection; they are not article-level
exclusions. Because only one 1,000-record page was retrieved for queries with
larger hit counts, this flow is **quota-oriented sampling**, not a comprehensive
PRISMA review of every hit. The raw queries and screening rows allow the work
to be re-run or extended without claiming unseen pages were reviewed.
