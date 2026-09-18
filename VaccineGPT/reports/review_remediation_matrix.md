# Strict review remediation matrix

This matrix records the implementation status of the supplied CNS/NSFC review.
Database release year is kept separate from paper publication year; a release
label is not presented as evidence that a paper was published recently.

| Review item | Decision | Implementation/evidence | Verification or limitation |
|---|---|---|---|
| Retire M1/M2 as the main architecture | Accepted | `vaccinegpt/` is the supported L0-L4 path; `src_m1` and `src_m2` moved to `legacy/` with deprecation notice. | Legacy import isolation and smoke checks required before release. |
| Complete the public-data path | Accepted | `vaccinegpt/real_data.py` and `scripts/train_five_layer_real.py` convert canonical sequence/label JSONL to the five-layer contract. | Requires the public JSONL artifacts to be downloaded/prepared locally. |
| Do not require all T1/T2/T3/T4 labels | Accepted | `task_mask` suppresses missing-task losses and the real-run report emits coverage. | T1/T4 remain unmeasured when their source assays are absent. |
| Treat missing T1/T4 as negative | Rejected | The adapter uses masked placeholders, not negative labels. | This is a data-integrity correction, not a biological claim. |
| Use graph evidence in the model | Accepted | `PPI_adj` and `node_x` are contract inputs; optional STRING edges are loaded by the adapter. | Identity adjacency is reported when no PPI file is supplied. |
| Require group-aware leakage control | Accepted in data contract | Existing grouped split utilities remain mandatory for prepared data; legacy fallback is no longer a supported path. | Homology clustering still requires MMseqs2/CD-HIT artifacts before a production benchmark. |
| Replace quadratic global ranking | Partially accepted | The five-layer path consumes masked task losses; production group/list batching remains a follow-up for large T4 candidate lists. | No unsupported ranking metric is reported for singleton groups. |
| Update source releases | Accepted | Manifest records VFDB 2025, IEDB 2024 update, DEG 15, STRING 2023 (v12.0), and OGEE v3. | OGEE/DEG provider export is still marked pending; no checksum is fabricated. |
| Claim 200/100 full-text evidence quotas | Rejected | QA remains failed when chapter quotas are not met; candidate metadata is not counted as full text. | The reports preserve this limitation. |

## Evidence interpretation

The review's criticism that a previous real-data command hard-required all four
tasks is correct and is fixed above. Its recommendation to treat OGEE/DEG
release labels as recent papers would be incorrect: the manifest records
database versions, while the literature pipeline independently records paper
dates and full-text status. Claims about biological mechanisms remain
regularizers or hypotheses unless directly supported by the evidence ledger.

## Executed public-data pilot

The current worktree downloaded the VFDB 2025 protein release, converted 64
records with `prepare_vfdb_records.py`, and completed a two-epoch CPU pilot.
The resulting report recorded T2 coverage 1.0 and T1/T3/T4 coverage 0.0; the
unobserved tasks were masked. This validates the data path only and is not a
held-out performance result.
