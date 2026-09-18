# VaccineGPT M1/M2 continuation record

## Confirmed objective

Build the M1 genome intelligent core (GIC) and M2 sequence design engine (SDE)
from the CLEF-derived scaffold. The implementation must be runnable with cached
features or synthetic data before external foundation-model weights and curated
biological datasets are available.

## Design decisions

- M1 exposes a stable `h_final` representation with shape `(batch, 128)`.
- SOM uses protein, DNA, and ncRNA views; INT additionally uses the omics view.
- External ESM-2, DNABERT-2, RNA-FM, ProteomeLM, and NetMHC adapters remain
  replaceable. They must not be represented by fabricated biological metrics.
- M2 consumes `h_final` and exposes T1/T2/T3 prediction heads plus T4 ranking.
- Every feature, label, checkpoint, and evaluation report carries the lineage
  tuple: model version, feature version, label version, and batch id.
- Train/validation/test boundaries must be established before fitting any
  preprocessing transform, and grouped splits must prevent locus/operon leakage.

## Current repository state before implementation

The existing scaffold had M1 and M2 smoke models, synthetic batches, lineage
helpers, a compatibility matrix, and an operations checklist. It did not yet
implement the complete SOM/INT contract, UDC validation, grouped splitting,
four-tier labels, full M2 task heads, ranking loss, statistical evaluation, or
persisted model/evaluation artifacts.

## Implementation record

This file is updated after each implementation phase. The next agent should
read it together with `README.md`, `shared/operations_checklist.md`, and
`configs/m1_m2_config.json` before changing interfaces.

## Pending execution checklist

- [x] Implement shared UDC schemas and validation.
- [x] Implement reproducible grouped data splitting and leakage checks.
- [x] Implement M1 SOM/INT-compatible encoders, attention fusion, relation-aware graph pass,
  and train/evaluate entry points.
- [x] Implement M2 four-tier labels, T1/T2/T3/T4 heads, multi-task training,
  and ranking evaluation.
- [x] Persist lineage-aware checkpoints and evaluation reports.
- [x] Add deterministic end-to-end synthetic validation.

## Validation record

- `scripts/run_m1_demo.py`: passed; loss decreased and lineage was emitted.
- `scripts/run_m2_demo.py`: passed; multi-task loss decreased and lineage was emitted.
- `scripts/run_end_to_end.py`: validates grouped splits, UDC feature records,
  M1-to-M2 tensor flow, multi-task loss, ranking loss, and checkpoint/report output.
- Latest synthetic run produced split sizes `66/12/18`, `h_final` shape
  `(96, 128)`, and a decreasing total loss from `2.6321` to `1.4858`.
- Generated artifacts: `artifacts/m1_m2_checkpoint.pt` and
  `artifacts/end_to_end_eval.json`.
- Added reusable T4 grouped NDCG/MRR metrics and pairwise ranking loss in
  `src_m2/metrics.py`; added T3 epitope aggregation and entropy interface in
  `src_m2/immune.py`.
- Added standalone focal and non-negative PU loss primitives in
  `src_m2/losses.py`; updated the operational checklist for the three-class T1
  and grouped T4 gates.
- Added lazy pretrained adapters and `configs/foundation_models.json` for
  ESM-2, CLEF checkpoints, DNABERT-2, ProteomeLM, Evo2, and configurable RNA-FM.
- Added `scripts/extract_foundation_features.py` for UDC-02 extraction and
  `scripts/train_from_udc.py` for direct M1+M2 training from real pooled
  features and UDC-03 labels.
- Added `configs/model_manifest.json` and `weights/README.md` to preserve
  upstream checkpoint provenance without copying large weights into the repo.
- Replaced the provisional RNA-FM Hugging Face path with the official
  `fm.pretrained.rna_fm_t12` adapter from `ml4bio/RNA-FM`; Evo2 remains an
  explicit alternative DNA backend rather than being mislabeled as RNA-FM.

## Latest validation

- Adapter factory imports without loading optional weights.
- Existing M1/M2 smoke and end-to-end validation still pass.
- `scripts/train_from_udc.py` trained successfully on a 30-gene UDC fixture,
  producing grouped split sizes `21/3/6`, a checkpoint, and a report.
- Added `scripts/download_public_data.py` with SHA256/version manifests,
  `scripts/prepare_dataset.py` for canonicalization, evidence-tier merging,
  exact deduplication, and stratified group splits, plus `scripts/build_graph.py`
  for leakage-safe seven-relation UDC-04 graphs.
- Added effective-number class weighting combined with label-tier reliability
  weighting so rare high-quality labels are retained without duplicating records.
- Added `configs/data_sources.json`, `shared/download.py`, and
  `scripts/download_public_data.py` for checksummed public-data acquisition.
- Added `shared/dataset.py`, `shared/split_manifest.py`,
  `scripts/prepare_dataset.py`, and `scripts/build_graph.py` for sequence
  canonicalization, exact deduplication, evidence merging, stratified
  group-level splits, and leakage-safe seven-relation graph construction.

## Explicit scope boundary

The current milestone validates the software contracts and train/evaluate wiring.
T3 provides a scalar prediction plus an epitope aggregation interface, while the actual
15-mer MHC/NetMHCIIpan feature adapter is intentionally deferred until curated
epitope data and the external predictor are supplied. T4 uses the stable
pairwise ranking interface for the same reason; its production NDCG calibration
must be run on grouped candidate lists rather than synthetic labels.
- [x] Update README and operations checklist with commands and M3-M6 handoff.

## Latest validation

- Adapter factory imports without loading optional weights and registers
  protein, DNA, RNA-FM, and Evo2 paths.
- Existing M1/M2 smoke and end-to-end validation pass.
- `scripts/train_from_udc.py` trained successfully on a 30-gene UDC fixture,
  producing grouped split sizes `21/3/6`, a checkpoint, and a report.
- The normalization fixture ran successfully and generated sequence, label,
  and split manifests. A fixture with fewer than two distinct biological
  groups is correctly rejected instead of fabricating a test set.

## Public-data boundary

The downloader automatically handles HTTP resources and the official NCBI
`datasets` CLI. Sources such as DEG/OGEE/DOOR2/VFDB exports, SRA raw reads,
and NetMHCIIpan require an exact release/export selection or a separate
bioinformatics workflow; they are represented as disabled/manual manifest
entries until their URLs, terms, and processing parameters are fixed.
The pipeline intentionally does not claim that downloading raw reads equals
having gene-level Tn-seq/CRISPRi measurements.

## Public-data pipeline implementation

- The source manifest attempts official NCBI, UniProt, STRING 2023 (v12.0),
  IEDB 2024 update, VFDB 2025 release files, DEG 15, OGEE v3, RegulonDB
  export, and SRA Toolkit downloads.
  Each result records status, byte count, SHA256, and errors.
- HTTP downloads are resumable through `.partial` files, retried, and checked
  for minimum size and obvious HTML/content mismatches.
- `scripts/audit_dataset.py` reports missing IDs, duplicate IDs/sequences,
  empty sequences, label/task distributions, failed downloads, and split
  overlap.
- `scripts/build_training_manifest.py` emits a no-duplication training
  manifest with effective-number and evidence-tier weights.
- `scripts/build_sra_tasks.py` emits explicit fastp/Bowtie2/TRADE-seq tasks;
  raw reads are not treated as gene-level labels until this processing step
  produces validated effects.
- Fixed grouped train/validation/test splits remain the primary report, with
  repeated grouped cross-validation reserved for small-data evaluation.

## 2026-09-15 public-data run

- Successfully downloaded and checksummed STRING E. coli links, VFDB SetA/SetB
  protein and nucleotide files, VFDB annotations, and a RegulonDB GFF3 export.
- Successfully retrieved 119,000 IEDB epitope records through the PostgREST
  endpoint using keyset pagination. The JSON cache is approximately 523 MB and
  has a recorded SHA256 in `data/metadata/download_manifest.json`.
- Normalization produced 144,715 unique sequence records and 144,715 reconciled
  label records after mapping duplicate sequence labels to canonical sequence
  representatives. The current supervised labels are T2 VFDB evidence and
  T3b IEDB evidence; T1 essentiality remains pending source-specific exports.
- The final grouped split contains 101,300/21,706/21,709 train/validation/test
  records with no ID overlap. `data/metadata/public_self_check.json` reports
  `pass`; `data/processed/public_final/training_manifest.json` is the training
  handoff.
- NCBI RefSeq, UniProt, and SRA were retried but remain explicitly failed:
  NCBI Datasets and SRA Toolkit are not installed in the current environment,
  while the UniProt endpoint timed out. OGEE/DEG remains pending because an
  authenticated or release-specific export is required. These are not silently
  represented as labels.
- The repository was migrated under `D:\Vaccine-GPT\Vaccine-GPT`; downloader
  paths are now anchored to the project directory rather than the caller's
  current working directory. Review documents were moved to the sibling
  top-level `reference/` directory.

## 2026-09-15 migration and retry update

- Added `scripts/check_data_artifacts.py` for post-migration artifact inventory
  and checksum validation.
- Fixed downloader edge cases: paged JSON finalization, relative path manifests,
  local tool resolution for NCBI `datasets` and SRA `prefetch`, and compressed
  tabular parsing in `scripts/prepare_dataset.py`.
- Download status after retries:
  - **Succeeded/cached**: IEDB epitope JSON, VFDB files, STRING links,
    RegulonDB gff3, RefSeq `assembly_summary_refseq.txt`, UniProt bacterial
    mirror (`uniprot_sprot_bacteria.dat.gz` via EBI).
  - **Still failing in this network**: UniProt REST stream/search endpoint
    (timeout/SSL EOF).
  - **SRA**: toolkit installed (`tools/sratoolkit.3.4.1-win64/bin/prefetch.exe`);
    one small E. coli accession (`DRR063436`) was started but stalled during
    HTTPS transfer in this environment.

## Weight and environment boundary

The code paths for all requested pretrained models are implemented and lazy,
but this Windows CPU workspace does not download or execute multi-gigabyte
weights. A production GPU/WSL environment must install `requirements-external`,
place the CLEF checkpoint at the configured local path, and record exact
checkpoint revisions/SHA256 values in `configs/model_manifest.json`. This is an
intentional reproducibility boundary, not a replacement for the upstream
weights.
