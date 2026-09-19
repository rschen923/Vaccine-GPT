# Vaccine-GPT

This repository hosts the L0-L4 five-layer Vaccine-GPT implementation.
The historical M1/M2 prototype is retained under `legacy/` for provenance only.

## Scope

- L0-L4: graph-aware representation, biological prediction, immune modeling,
  attenuation ranking, and reproducible training contracts
- Explicit task masks for partially observed public labels
- Shared lineage and operational contracts

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the five-layer smoke test:
   ```bash
   python train.py --smoke
   ```
4. Run the mathematical and contract checks:
   ```bash
   python verification\verify_math.py
   python verification\contract.py
   ```
5. Run a real-data pilot after canonical JSONL preparation:
   ```bash
   python scripts\train_five_layer_real.py `
     --sequences data\processed\public_final\sequences.jsonl `
     --labels data\processed\public_final\labels.jsonl
   ```

## Directory structure

- `vaccinegpt/` — authoritative L0-L4 model, contract, losses, and trainer
- `legacy/` — deprecated M1/M2 source retained for reproducibility; not a
  supported training entry point
- `shared/` — lineage metadata and operation checklist
- `shared/contracts.py` and `shared/splits.py` — UDC checks and grouped leakage-safe splits
- `configs/` — model and runtime configuration
- `scripts/` — runnable smoke-test entry points
- `ops/` — operations and release checklist

## Fallback strategy for external models

If a view depends on external pretrained encoders such as ESM-2, DNABERT-2, RNA-FM, or ProteomeLM and those dependencies are unavailable in the local environment, the project follows a cache-first design:

- store extracted features in a cache,
- keep the downstream M1/M2 interface stable,
- isolate the training logic behind a clear script-level comment,
- preserve the same schema so later stages can resume the pipeline without reworking interfaces.

This is deliberately documented in the code-level comments near each entry point.

## Notes

The five-layer path is a working research implementation, not a claim of
clinical or biological validity. Every report must include label coverage,
provenance, and the distinction between measured labels and masked tasks.

The smoke path intentionally uses deterministic synthetic data. Its metrics
validate tensor shapes and loss plumbing only. The real-data adapter consumes
canonical public JSONL records and reports missing T1/T4 assays with masks; it
never turns missing labels into negatives.

## Real pretrained-model workflow

`configs/foundation_models.json` records the pretrained starting points. The
adapters load original upstream weights and freeze them by default; five-layer
projections, graph layers, and task heads are the trainable parts. Set `freeze`
to `false`, or use `unfreeze_last_n_layers`, only for deliberate fine-tuning.
CLEF is loaded from a local checkpoint and can be chained after the ESM-2 token
representation; ProteomeLM is optionally concatenated into the protein view.

After installing the optional production dependencies, extract canonical records:

```powershell
python legacy/scripts/extract_foundation_features.py `
  --config configs/foundation_models.json `
  --input data/sequences.jsonl `
  --output features/FEAT.1.0.0/features.jsonl `
  --lineage lineage/lineage.json
```

The old UDC M1/M2 command remains available only under `legacy/`; new training
uses the five-layer contract:

```powershell
python scripts\train_five_layer_real.py `
  --sequences data\processed\public_final\sequences.jsonl `
  --labels data\processed\public_final\labels.jsonl `
  --output checkpoints\real_five_layer
```

The former UDC M1/M2 command remains at
`legacy/scripts/train_from_udc.py` for historical reproduction only.

Evo2 is supported as a DNA adapter. To use it instead of DNABERT-2, change the
`dna.backend` field in `configs/foundation_models.json` to `evo2` and set its
`model_id`/`layer_name`; the downstream feature contract is unchanged. It is
not silently treated as an RNA model:
RNA-FM remains the preferred RNA adapter and its checkpoint ID is configurable
because the referenced example repository is not itself a stable model
distribution.

## Public-data acquisition and preparation

`configs/data_sources.json` is a versioned source manifest. It supports the
official NCBI `datasets` CLI, resumable HTTP downloads, VFDB/IEDB/STRING
release endpoints, and SRA Toolkit accessions. It computes SHA256 checksums,
rejects suspicious HTML/empty responses, and writes
`data/metadata/download_manifest.json`. Authentication, license, endpoint, or
resource failures are recorded explicitly as failed results.

Large downloaded raw datasets are intentionally excluded from Git tracking.
Their URLs, checksums, byte counts, and failure status remain in
`data/metadata/download_manifest.json`; rerun the downloader to restore them.
After moving the workspace, verify every recorded local artifact with:

```powershell
python scripts/check_data_artifacts.py
```

```powershell
python scripts/download_public_data.py --config configs/data_sources.json
```

Prepare downloaded tables after mapping their columns to the canonical aliases
(`gene_id`, `protein_sequence`/`dna_sequence`, `task`, `label`, `level`,
`operon_id`/`contig_id`):

```powershell
python scripts/prepare_dataset.py `
  --sequences data/raw/sequences.tsv `
  --labels data/raw/essentiality.tsv data/raw/vfdb.tsv data/raw/iedb.tsv `
  --lineage lineage/lineage.json `
  --output-dir data/processed
```

The preparation step canonicalizes sequences, removes exact duplicates,
merges conflicting evidence by tier (L1 > L2 > L3 > L4), writes UDC-compatible
records, and creates a stratified group split. It never up-samples duplicate
biological records. Instead, training can use effective-number class weights
and tier reliability weights so rare but high-quality L1 examples are not
discarded or drowned by L4 pseudo-labels.

For source-specific evidence mapping, use the label builder. Unmappable rows
are retained in a rejected-record file and duplicate evidence is merged by
task and tier:

```powershell
python scripts/build_labels.py `
  --input data/raw/essentiality.jsonl --task T1 --level L1 `
  --input data/raw/vfdb.jsonl --task T2 --level L1 `
  --input data/raw/iedb.json --task T3 --level L2 `
  --lineage lineage/lineage.json `
  --output data/processed/labels.jsonl
```

Audit normalized outputs and build a training manifest:

```powershell
python scripts/audit_dataset.py `
  --inputs data/processed/sequences.jsonl data/processed/labels.jsonl `
  --download-manifest data/metadata/download_manifest.json `
  --split-manifest data/processed/split_manifest.json

python scripts/build_training_manifest.py `
  --sequences data/processed/sequences.jsonl `
  --labels data/processed/labels.jsonl `
  --splits data/processed/split_manifest.json `
  --output data/processed/training_manifest.json
```

For SRA experiments, create explicit processing tasks after raw reads and a
reference index are available:

```powershell
python scripts/build_sra_tasks.py `
  --accessions data/metadata/sra_accessions.txt `
  --reference references/pathogen_index
```

## New L0--L4 architecture

The implementation in `vaccinegpt/` is the only supported mainline. Historical
`src_m1/` and `src_m2/` are moved under `legacy/` and are not imported by the
mainline. The five-layer path wires the specified VIB latents,
L1--L4 physiological modules, detached layer coupling, analytical consistency
losses, and T1/T2/T3/T4 task objectives. Synthetic data is exclusively for
software smoke testing and does not provide biological-performance evidence.

### Training data contract

`--data` accepts a JSONL file containing exactly one batch object. Array
nesting represents tensor dimensions. Every field in
`vaccinegpt.contracts.REQUIRED_FIELDS` is mandatory and finite, including
`x_a (B,L,64)`, `x_b (B,32)`, `pcd_A (B,4)`, `ch_covariates (B,16)`,
`topo_feat (B,4)`, PPI tensors, PU indices, HLA frequencies, peptide/pseudo
features, labels, and population observations. The loader explicitly rejects
missing or malformed values; data adapters must derive fields before training.
Shard multi-batch data and train each shard independently.

From `VaccineGPT/`, prepare the environment with `pip install -r
requirements.txt`, then run:

```powershell
python train.py --smoke
python train.py --epochs 8 --stage-epochs 1
python train.py --data path\to\contract_batch.jsonl --epochs 8 --stage-epochs 1 --checkpoint checkpoints\five_layer.pt
python verification\verify_math.py
python verification\verify_l4.py
```

The curriculum is L0, L1, L2, L3, L4, then joint training. Inter-layer states
are detached, and the primary L3-to-L4 route is `tau_TI -> v0`; alpha coupling
is disabled by default. Euler is the default differentiable ODE backend
(`steps=1000`, `dt=0.02`); `dopri5` needs optional `torchdiffeq` and reports a
clear installation error when absent.

## Evidence and report pipeline

The research pipeline uses the date window `2024-09-19` through `2026-09-19`
and records OpenAlex, PubMed, Europe PMC, and Crossref provenance. It does not
count metadata candidates as reviewed evidence:

```powershell
python scripts\research\literature_pipeline.py self-test
python scripts\research\literature_pipeline.py retrieve configs\literature\query_manifest.json literature-output
python scripts\research\literature_pipeline.py compact literature-output\papers.jsonl
python scripts\research\literature_pipeline.py screen-pending literature-output
python scripts\research\literature_pipeline.py screen-fulltext literature-output --per-section 20
python scripts\research\literature_pipeline.py export-citations literature-output reports\references_gbt7714.txt
python scripts\research\literature_pipeline.py qa configs\literature\query_manifest.json literature-output
python scripts\research\generate_reports.py --output reports --literature literature-output
```

`screen-pending` deliberately marks metadata-only records as `uncertain`.
Full-text eligibility, evidence extraction, contradiction review, and formula
proof classification must be completed before any paper is counted as
included. The generated status report therefore reports the current retrieval
count separately from the included-evidence count and fails quotas honestly.
See `docs\research_pipeline.md` and `reports\04_formula_ledger_template.md`.

## Research skills and project analysis

The distilled research workflows are in the repository-level `../skill/`
directory: literature retrieval and download, evidence-aware review writing,
NSFC project design, academic Markdown/PPT/figure generation, and an
integrated resumable pipeline. Validate their required quality gates with:

```powershell
python scripts/validate_research_skills.py
```

The current academic synthesis and experiment plan are:

- `../reference/VaccineGPT_学术综述与项目总分析.md`
- `../reference/实验设计_E_coli_E_piscicida_组学闭环.md`

Run the data self-correction gate before training:

```powershell
python scripts/self_check.py `
  --sequences data/processed/public_final/sequences.jsonl `
  --labels data/processed/public_final/labels.jsonl `
  --splits data/processed/public_final/split_manifest.json `
  --output data/metadata/public_self_check.json
```

Build the seven-relation graph after producing relation TSVs:

```powershell
python scripts/build_graph.py `
  --nodes data/processed/sequences.jsonl `
  --edges data/raw/string.tsv data/raw/door2.tsv data/raw/eggnog.tsv `
  --splits data/processed/split_manifest.json `
  --lineage lineage/lineage.json `
  --output data/processed/graph.json
```
