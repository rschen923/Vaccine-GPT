# VaccineGPT — Evidence-Aware Vaccine Target Design Pipeline

> **Status:** Phase implementation; not yet validated with biological gold labels or experimental anchors.  
> **Generated:** 2026-10-07 | **Repository:** `rschen923/Vaccine-GPT`

VaccineGPT is a reproducible, evidence-aware computational pipeline that takes annotated bacterial protein sequences as input and produces ranked attenuation targets, subunit vaccine candidates, and full AEP (Actionable Evidence Package) reports. It implements a three-stage architecture:

1. **M1 Encoding** — Aligns sequence/structure representations with evolutionary and fitness measurements via cross-modal contrastive learning.
2. **GCE (Graph Context Encoder)** — Builds genomic context graphs (operon, co-expression, co-adaptation, interaction edges) using GATv2.
3. **M2 Decision Kernel** — Produces calibrated predictions across four heads (H1–H4), conformal intervals, Pareto fronts, and safety flags.

A companion module, **LAV-Net**, provides independent attenuation-target suitability ranking for live attenuated bacterial vaccine design using CMCL + ATD + IGP stages.

---

## Architecture Overview

```
v1: Annotated protein sequences (NOT DNA)
  ├─ Frozen ESM-2 → residue representation [B,L,1280], L≤256
  ├─ CLEF-style Encoder A → z_pub [B,128] + positional attention
  ├─ Public modalities (text/structure/MSA/PSSM) → independent projection heads
  │    └─ Contrastive loss (InfoNCE)
  ├─ Private omics (PACE 3 + RNA-seq 8 + optional ECP 3) → private adapters
  │    └─ Negative sampling within same genome only; missing ≠ zero
  └─ z_pub + private context/masks → 2-layer GATv2 GCE → z_gctx
       └─ M2 decision kernel: H1/H2/H3/H4 + conformal intervals + Pareto → AEP

v2: Whole-genome DNA windows
  └─ Evo2 embedding → element annotation → extended GCE nodes/edges → H1-H4 frozen reuse + H5 head
       └─ AEP v2 maintains same nine-field contract

M3-S / M3-N: Read-only M1/M2 and AEP contracts; no back-propagation to mainline.
```

### Four Axioms

| Axiom | Implementation |
|---|---|
| **A0 Traceability** | Every label retains source/row-level provenance; model/feature/label/batch versions tracked via lineage. Unknown values use `null`, `unavailable`, or `E4` — never random scores. |
| **A1 Single Mainline** | v1 mainline: `protein → M1 → GCE → H1–H4 → M2/AEP`. M3-S/N are read-only; v2 adds DNA window + H5. |
| **A2 Data Hierarchy** | E1/E2 labels serve as gold supervision; E3/E4 retained as weak supervision; conflicting evidence quarantined. |
| **A3 Completeness** | AEP must include uninterpreted items, warnings, and external parameter status. No ranking without valid evidence. |

---

## Project Layout

```text
Vaccine-GPT/
├── VaccineGPT/                     # Main project code
│   ├── src_m1/                     # M1 encoding: encoders, GCE, training
│   │   ├── encoders/               # Foundation model adapters (ESM-2, ESM-3, Evo2)
│   │   ├── models.py               # CLEF-style Transformer, InfoNCE losses
│   │   ├── gce.py                  # GATv2 graph context encoder
│   │   ├── graph.py                # Graph construction & metrics
│   │   ├── genome_v2.py            # v2 DNA window schema
│   │   └── pipeline.py             # End-to-end M1→M2 wiring
│   ├── src_m2/                     # M2 decision kernel
│   │   ├── models.py               # H1–H4 task heads
│   │   ├── h5.py                   # v2 H5 head scaffold
│   │   ├── decision.py             # PACE, conformal intervals, Pareto, ACQ
│   │   ├── aep.py                  # Actionable Evidence Package output
│   │   ├── losses.py               # Multi-head loss functions
│   │   └── metrics.py              # Evaluation metrics
│   ├── src_m3/                     # M3 read-only interfaces
│   │   └── interfaces.py           # M3-S (subunit) / M3-N (nucleic acid) contracts
│   ├── shared/                     # Cross-module utilities
│   │   ├── dataset.py              # Dataset loading & splitting
│   │   ├── data_quality.py         # Evidence-tier auditing & cleaning
│   │   ├── lineage.py              # Version tracking & provenance
│   │   ├── self_correction.py      # Iterative correction loops
│   │   └── udc_io.py               # Unified Data Contract I/O
│   ├── scripts/                    # Training, extraction, and pipeline scripts
│   │   ├── extract_esm2_embeddings.py
│   │   ├── train_m1_contrastive.py
│   │   ├── train_real_pipeline.py
│   │   ├── run_end_to_end.py
│   │   ├── audit_and_clean_data.py
│   │   └── ... (30+ scripts)
│   ├── tests/                      # Unit and smoke tests
│   │   ├── test_architecture.py
│   │   └── ...
│   ├── configs/                    # Model manifests, data sources, weights
│   ├── data/                       # Raw/processed data (gitignored)
│   ├── weights/                    # Pretrained model weights (gitignored)
│   ├── third_party/                # Licensed third-party code (CLEF)
│   ├── requirements.txt            # Core dependencies
│   ├── requirements-esm3.txt       # ESM-3 runtime dependencies
│   └── requirements-external.txt   # External tool dependencies
├── VaccineGPT/LAV-Net/             # Attenuation target ranking pipeline
│   ├── lavnet/                     # Model, labels, metrics
│   ├── scripts/                    # Downloads, labeling, training
│   ├── configs/                    # Label sources, public data manifests
│   ├── docs/training_flow.md       # Three-stage training documentation
│   └── tests/                      # Self-test suite (~23 checks, ~13s CPU)
├── reference/                      # Research reports, iteration logs, evidence
│   ├── iterations_261007/          # 10 rounds of literature campaigns
│   ├── VaccineGPT_架构报告_261007.md
│   ├── VaccineGPT_训练报告_261007.md
│   └── ...
├── archive/                        # Archived materials
└── README.md                       # This file
```

---

## Foundation Models

| Track | Model | Source | Role | Trainable? |
|---|---|---|---|---|
| **v1 Protein** | ESM-2 (650M) | `facebookresearch/esm` | Frozen residue representation | ❌ Frozen |
| **v1 Protein** | CLEF Encoders | `PengYueECUST1997/CLEF` | T3SS/T4SS/T6SS effector prediction | ✅ Lightweight projections |
| **v1 Protein** | ESM-3 (open small) | `Biohub/esm` | Optional multi-modal adapter | ❌ Frozen backbone |
| **v2 DNA** | Evo2 (7B) | `ArcInstitute/evo2` | Frozen DNA-window embedding | ❌ Frozen |
| **v2 DNA** | DNABERT-2 | `MAGICS-LAB/DNABERT_2` | Optional DNA baseline | ❌ Frozen |
| **ncRNA** | RNA-FM | `ml4bio/RNA-FM` | Optional ncRNA features | ❌ Frozen |

> **Note:** ESM-3, Evo2, and CLEF checkpoints require separate installation. See `requirements-esm3.txt` and individual model licenses.

---

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

For ESM-3 support:

```bash
python -m pip install -r requirements-esm3.txt
```

### 2. LAV-Net Quick Start (standalone)

```powershell
cd VaccineGPT\LAV-Net
python -m pip install -e .
python scripts\download_public_data.py --manifest configs/public_sources.json
python scripts\build_labels.py --config configs/label_sources.json
python -m pytest -q
python tests\test_lavnet.py        # ~23 checks, ~13s on CPU
```

### 3. Full Pipeline (M1 → M2 → AEP)

```bash
# Extract embeddings from frozen foundation model
python scripts/extract_esm2_embeddings.py

# Build labels and audit data quality
python scripts/build_labels.py
python scripts/audit_and_clean_data.py

# Run M1 contrastive training
python scripts/train_m1_contrastive.py

# Run end-to-end pipeline
python scripts/run_end_to_end.py
```

---

## M1 Tensor Contracts

| Key | Shape | Meaning |
|---|---|---|
| `seq` | `[B, L, 1280]` | ESM-2 residue representation |
| `evo` | `[B, 64]` | Evolutionary features (conservation 32 + dipeptide 32) |
| `fit` | `[B, 17]` | Tn-seq fitness profile (17 conditions) |
| `ctx` | `[B, 16]` | Genomic context (operon, strand, spacing, length) |
| `expr` | `[B, 4]` | Expression features |
| `y` | `[B]` | Label |

---

## M2 Task Heads

| Head | Role | Supervision |
|---|---|---|
| **H1** | In vivo colonization / essentiality | E1/E2 gold (currently unavailable) |
| **H2** | Surface localization / antigen evidence | E1/E2 gold (currently unavailable) |
| **H3** | Effector secretion | E3 predictive (CLEF/STRING) |
| **H4** | In vitro core essentiality | E4 weak labels |

Each head is a linear logit on the shared 128-dim `z_gctx` input. When corresponding assay/gold labels are absent, the head's supervision state is marked `unavailable` — unannotated genes are **never** silently converted to negative labels.

---

## Output Contract: AEP (Nine Fields)

Every candidate row in the Actionable Evidence Package contains:

```
gene/product | Att interval | Ret | Safe | evidence E1–E4 | three-reader | mechanism | warning | implementation path
```

- **Att interval**: Conformal prediction interval for attenuation suitability
- **Ret**: Retention score (surface expression × neighborhood antigen fraction × centrality)
- **Safe**: Safety flag (four safety indicators, hard veto if H4 > 0.5)
- **Three-reader**: Sequence attention ranking, private spatial prototype ranking, log-contribution of Att/Ret/Safe
- **implementation_path**: Always empty in v1; requires experimental build data for training

---

## Safety and Evidence Rules

1. **Unknown remains unknown** — Missing labels stay `null`; never auto-converted to negatives.
2. **Homology-based splits** — Train/validation/test splits defined by MMseqs2 ≥90% identity clusters or pre-registered genome holdouts.
3. **Hard exclusion gates** — High-risk antigens excluded by IGP gate, not traded off in loss.
4. **Threshold calibration** — Validation thresholds selected and saved before test scoring.
5. **Provenance tracking** — Every download recorded with URL, retrieval time, SHA-256, access state, and license.
6. **No false claims** — Model predictions are not experimental evidence. All outputs carry appropriate uncertainty markers.

---

## Current Status & Limitations

| Item | Status | Reason |
|---|---|---|
| v1 rejects DNA input | ✅ Code contract enforced | M1 receives residue embeddings; UDC feature contract blocks DNA |
| M1/GCE/M2 tensor contracts | ✅ Smoke tested | 20 unit + synthetic forward passes pass |
| 1,000 real protein records through pipeline | ✅ Engineering pass | One-hot self-supervised M1 trained; M2 forward works; 1,000 rows remain `unranked` |
| AEP four-piece set with unknown states | ✅ Schema ready | M2 has no E1/E2 gold training; gates keep all rows `unranked` |
| G-A/B/C/D, A1–A9 validation | ❌ Not verified | No formal CLEF reproduced metrics, EIB202 anchors, blind/cali sets |
| Full supervised training ready | ❌ Blocked by data gate | E1/E2 gold labels = 0 in current public-final data |
| MMseqs2 90% homology clustering | ❌ Not run | Exact duplicates and homologous leakage are distinct concerns |

**Machine execution boundary:** Current environment is CPU-only PyTorch. The T600 4GB GPU is detected but not usable with the installed CPU PyTorch. ESM-2 650M, CLEF checkpoints, and Evo2 dependencies are not installed locally.

---

## Research Campaigns

The project includes a structured literature evidence campaign system:

- **10 rounds** completed, each fetching and validating 200 open-access full-text citations from Europe PMC
- **202 code repositories** inspected (192 globally unique), with README/implementation/license status
- **93 campaign works** reviewed by AI sub-agents (89 unique), with 90 random citation lines sampled for verification
- Round 5 review blocked by safety filter; remaining 9 rounds' 90 excerpts reviewed via AI sub-agent excerpt analysis

See `reference/iterations_261007/` for per-round dossiers and manual review logs.

---

## License

- **Main code:** Repository-specific license (see LICENSE file)
- **CLEF third-party code:** MIT (see `third_party/CLEF_LICENSE.txt`)
- **Foundation model checkpoints:** Subject to individual model licenses (ESM-2: Apache 2.0; Evo2: see Arc Institute terms)
- **Model predictions are NOT experimental evidence** — they are computational hypotheses requiring wet-lab validation.

---

## Citation

If you use VaccineGPT in your research, please cite the original foundation models:

```bibtex
@article{esm2,
  title={Language models generalize beyond proteins},
  author={Lin, Zhenlin and Akin, Halil and Rao, Roshan and others},
  journal={Nature Methods},
  year={2023}
}

@article{evo2,
  title={Evo2: A universal sequence model for genomics},
  author={Arc Institute},
  year={2024}
}
```

---

## Contributing

This is a research implementation. Contributions welcome for:
- Adding new foundation model adapters
- Improving data quality audits
- Extending LAV-Net label sources
- Bug fixes and test coverage

Please ensure all new code follows the four axioms (traceability, single mainline, data hierarchy, completeness).
