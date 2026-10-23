# LAV-Net

LAV-Net is a reproducible, evidence-aware pipeline for ranking attenuation
targets for live attenuated bacterial vaccine design. It implements the three
stage design in the accompanying proposal:

1. **CMCL** aligns sequence/structure representations with evolutionary and
   fitness measurements.
2. **ATD** predicts attenuation-target suitability.
3. **IGP** is an independent immunogenicity guard; known/high-risk antigens
   are excluded by a hard gate rather than traded off in a loss.

The three are chained independent networks rather than heads of one network:
their label volumes differ by an order of magnitude, so a shared trunk would
let the higher-volume task dominate the gradient.

The repository is a software and data-preparation implementation. It does not
claim wet-lab efficacy, and it never turns an antigen score into proof that
gene deletion will reduce protection.

## Layout

```text
LAV-Net/
  lavnet/                 model, data contracts, labels, evaluation
  scripts/                downloads, labeling, training and ranking
  configs/                versioned source and model configuration
  docs/                   training flow and implementation notes
  tests/                  unit and smoke tests
  data/
    raw/                  downloaded public data (gitignored)
    processed/            derived datasets (gitignored)
    manifests/            provenance and checksums (tracked)
```

## Model

`lavnet/model.py` holds the three networks plus the shared primitives
(`FrozenPCA`, `Adapter`, `ProjHead`, `cosine_warmup`, `count_params`). Every
width comes from `CMCLConfig` / `ATDConfig`; the modules hold no dimension
constants, so a feature-pipeline change fails at construction time instead of
midway through training.

| Network | Class | Role | Trainable params |
|---|---|---|---|
| Cross-modal contrast | `CMCL` | aligns evolutionary and fitness evidence with the sequence representation | 10,016 |
| Attenuation target | `ATD` | probability that a gene is usable as an attenuation target | 4,354 |
| Immunogenicity guard | `IGP` | probability that the gene product is a major antigen; used as a hard mask | 801 |

`CMCL` parameter count scales with the condition count; the value above is for
17 Tn-seq conditions.

The ESM-3 backbone is deliberately outside this package. Callers run it once,
cache the pooled features, and feed them in; recomputing 3,612 sequences per
training run would make cross-validation and ablations impractical.

| Key | Shape | Meaning |
|---|---|---|
| `seq` | `[B, 1536]` | ESM-3 sequence-track pooled representation |
| `str` | `[B, 4096]` | ESM-3 structure-track pooled representation, may be `None` |
| `evo` | `[B, 64]` | evolutionary features (conservation spectrum 32 + dipeptide composition 32) |
| `fit` | `[B, 17]` | Tn-seq fitness profile (17 conditions) |
| `tokens` | `[B, L, 32]` | residue-level tokens for ATD, may be `None` |
| `ctx` | `[B, 16]` | genomic context (operon position, strand, spacing, length) |
| `expr` | `[B, 4]` | expression features |
| `y` | `[B]` | label |

`FrozenPCA` starts from a random orthogonal basis so a forward pass runs.
Formal training must replace it with `FrozenPCA.from_matrix(...)` using
components fitted on the training fold only, and the swap has to happen before
training starts — otherwise the whole run optimises inside a data-independent
subspace. `IGP` returns both `logit` and `prob`: losses are computed on the
logit scale, threshold calibration on the probability scale.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
python scripts\download_public_data.py --manifest configs/public_sources.json
python scripts/build_labels.py --config configs/label_sources.json
python -m pytest -q
python tests\test_lavnet.py
```

`tests\test_lavnet.py` is a standalone 23-check self-test (model shapes,
contrastive loss descent, ATD learning, ablation) that needs only PyTorch and
NumPy and finishes in about 13 s on CPU. The three-stage training loop is in
[docs/training_flow.md](docs/training_flow.md).

Large raw sequencing files are opt-in:

```powershell
python scripts/download_public_data.py --manifest configs/public_sources.json --include-raw-reads
```

The default download retrieves source metadata, reference proteome records,
and published tabular/annotation resources. Every item is recorded with URL,
retrieval time, SHA-256, access state, and license/usage notes.

## Safety and evidence rules

- Unknown or missing labels remain unknown; they are never silently converted
  to negative labels.
- Homology groups, not individual proteins, define train/validation/test
  splits.
- IGP is a hard exclusion gate. A high IGP score means “reported antigen
  risk”, not “deletion is experimentally prohibited”.
- Thresholds are selected on validation evidence and saved before test scoring.
- Public data are downloaded only from provider endpoints named in the
  manifest; no credentials or private data are required.

