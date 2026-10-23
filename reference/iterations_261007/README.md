# GCDTO 261007 iteration index

**Retrieval and repository-inspection quotas passed; full-corpus claim review did not.**

For each topic, the `automated_rounds/iteration_NN.md` dossier contains the
baseline claim, themed OA full-text reference count, candidate support/counter
sources, synthesis, falsifiable test, and a 200-entry inline full-text reference
index. Every round has 200 records whose Europe PMC fullTextXML payload hashes
were verified.

| Round | Topic | OA full-text references | Public code repositories | Random AI-assisted excerpt review |
|---:|---|---:|---:|---|
| 1 | Protein foundation models | 200 | 20 | 10 selected papers reviewed |
| 2 | DNA/genome foundation models | 200 | 20 | 10 selected papers reviewed |
| 3 | Multimodal/contrastive protein and omics | 200 | 20 | 10 selected papers reviewed |
| 4 | Bacterial vaccine/reverse vaccinology | 200 | 20 | 10 selected papers reviewed |
| 5 | Bacterial functional genomics/fitness | 200 | 20 | **Blocked** by safety filter; not rerouted |
| 6 | Bacterial genome annotation/ORF | 200 | 20 | 10 selected papers reviewed |
| 7 | Graph learning in biology | 200 | 20 | 10 selected papers reviewed |
| 8 | Calibration/conformal/uncertainty | 200 | 20 | 10 selected papers reviewed |
| 9 | Pareto/multiobjective/active learning | 200 | 22 | 10 selected papers reviewed |
| 10 | Sequence QC/homology/leakage | 200 | 20 | 10 selected papers reviewed |

The article packets are available at
`..\data\research_campaign_261007\review_packets\round_NN_sample.jsonl`;
normalized work metadata, abstracts, evidence locations, license status, and
raw-response/full-text hashes are in
`..\data\research_campaign_261007\normalized_evidence_ledger.jsonl`. The full
OA XML snapshots are compressed in that external data tree, not copied into
the Git repository.

## Per-round dossiers

| # | Automated research dossier | Sampled full-text review | Codebase review |
|---:|---|---|---|
| 1 | `automated_rounds/iteration_01.md` | `iteration_01_manual_review.md` | `iteration_01_code_review.md` |
| 2 | `automated_rounds/iteration_02.md` | `iteration_02_manual_review.md` | `iteration_02_code_review.md` |
| 3 | `automated_rounds/iteration_03.md` | `iteration_03_manual_review.md` | `iteration_03_code_review.md` |
| 4 | `automated_rounds/iteration_04.md` | `iteration_04_manual_review.md` | `iteration_04_code_review.md` |
| 5 | `automated_rounds/iteration_05.md` | `iteration_05_manual_review_blocked.md` | `iteration_05_code_review.md` |
| 6 | `automated_rounds/iteration_06.md` | `iteration_06_manual_review.md` | `iteration_06_code_review.md` |
| 7 | `automated_rounds/iteration_07.md` | `iteration_07_manual_review.md` | `iteration_07_code_review.md` |
| 8 | `automated_rounds/iteration_08.md` | `iteration_08_manual_review.md` | `iteration_08_code_review.md` |
| 9 | `automated_rounds/iteration_09.md` | `iteration_09_manual_review.md` | `iteration_09_code_review.md` |
| 10 | `automated_rounds/iteration_10.md` | `iteration_10_manual_review.md` | `iteration_10_code_review.md` |

**Coverage boundary:** 2,000 OA full-text records were retrieved and checksum
validated; AI subagents reviewed supplied abstracts and selected sections from
90 randomly selected round-reference rows (89 unique works). Six prior
source reviews add four campaign works outside this random sample and two
arXiv works outside the campaign, for 93 campaign works and 95 unique
sampled-section reviews overall. The other 1,713 campaign works did not receive
individual claim review. This is not equivalent to a human manually reading all
2,000 papers. Round 5’s AI-assisted review was blocked by the safety filter and was
not attempted via another route. Only the sampled review files may be cited as
AI-assisted section-level evidence. The other works are accessible
full-text references with automated title/abstract screening and section
indexes, pending deeper review.

**Code quota:** 202 round-repository references and 192 globally unique public
repositories were recorded. License status refers to inspected repository
files, not a legal review of all weights, datasets, and dependencies.
