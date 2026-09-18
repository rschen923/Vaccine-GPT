# Vaccine-GPT

This repository hosts the M1 (GIC) and M2 (SDE) implementation scaffold for the Vaccine-GPT project.

## Scope

- M1: multi-view representation learning and integration
- M2: multi-task prediction + ranking for vaccine candidate evaluation
- Shared lineage and operational contracts for later modules

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the M1 smoke demo:
   ```bash
   python scripts/run_m1_demo.py
   ```
4. Run the M2 smoke demo:
   ```bash
   python scripts/run_m2_demo.py
   ```
5. Run the deterministic end-to-end M1 -> M2 validation:
   ```bash
   python scripts/run_end_to_end.py
   ```

## Directory structure

- `src_m1/` — M1 model logic, training, and evaluation
- `src_m1/encoders/` — lazy adapters for ESM-2, CLEF, DNABERT-2, ProteomeLM,
  Evo2, and configurable RNA-FM-compatible checkpoints
- `src_m1/pipeline.py` — real-sequence foundation-model to M1 path
- `src_m2/` — M2 model logic, training, and evaluation
- `src_m2/metrics.py` and `src_m2/immune.py` — grouped ranking and T3 aggregation utilities
- `src_m2/losses.py` — focal and non-negative PU loss primitives
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

This is a working prototype scaffold designed for early validation and extension, not a complete production pipeline.
It is meant to give the project a clean foundation for the subsequent modules while preserving compatibility with the CLEF-derived design logic.

The end-to-end script intentionally uses deterministic synthetic data. Its metrics
validate tensor shapes, train/validation/test isolation, loss plumbing, artifact
writing, and M1-to-M2 interfaces; they are not biological performance claims.

For production data, replace the synthetic feature provider with UDC-02 cached
records and provide grouped candidate IDs for T4. T3's `protein_immunogenicity`
expects calibrated epitope scores; it does not call NetMHCIIpan or infer those
scores without an explicit external adapter.

## Real pretrained-model workflow

`configs/foundation_models.json` records the pretrained starting points. The
adapters load original upstream weights and freeze them by default; M1
projections, graph layers, and M2 heads are the trainable parts. Set `freeze`
to `false`, or use `unfreeze_last_n_layers`, only for deliberate fine-tuning.
CLEF is loaded from a local checkpoint and can be chained after the ESM-2 token
representation; ProteomeLM is optionally concatenated into the protein view.

After installing the optional production dependencies, extract UDC-02 records:

```powershell
python scripts/extract_foundation_features.py `
  --config configs/foundation_models.json `
  --input data/sequences.jsonl `
  --output features/FEAT.1.0.0/features.jsonl `
  --lineage lineage/lineage.json
```

Once UDC-02 pooled features and UDC-03 labels have been exported, train both
modules without changing model code:

```powershell
python scripts/train_from_udc.py `
  --features features/FEAT.1.0.0/features.jsonl `
  --labels labels/LABEL.1.0.0/labels.jsonl `
  --genes data/gene_ids.json `
  --track SOM `
  --lineage lineage/lineage.json `
  --output checkpoints/real_m1_m2.pt
```

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

The new implementation in `vaccinegpt/` leaves historical `src_m1/` and
`src_m2/` untouched. It wires the specified CLEF dual encoder, VIB latents,
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
