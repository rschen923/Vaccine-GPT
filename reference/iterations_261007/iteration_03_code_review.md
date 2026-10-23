# GCDTO 迭代 03 — Multimodal / contrastive biological code 核验

**已检查 20/20 个公共、含代码 GitHub repositories**。清单含直接 sequence–text、sequence–structure 与 adjacent multi-omics 工具；不是 20 个同一种 contrastive architecture。

| # | Repository | modalities / path | License 状态 |
|---:|---|---|---|
| 1 | [DeepGraphLearning/ProtST](https://github.com/DeepGraphLearning/ProtST) | seq+biomedical text; `protst/model.py` | Apache-2.0 |
| 2 | [PengYueECUST1997/CLEF](https://github.com/PengYueECUST1997/CLEF) | ESM2 + biological features; `src/CLEF.py` | MIT |
| 3 | [westlake-repl/ProTrek](https://github.com/westlake-repl/ProTrek) | sequence/structure/function-text tri-modal; `model/ProTrek/abstract_model.py` | LICENSE present, exact SPDX not checked |
| 4 | [diaoshaoyou/ProtCLIP](https://github.com/diaoshaoyou/ProtCLIP) | sequence+description; `protclip/pretrain.py` | Apache-2.0 |
| 5 | [instadeepai/bioclip](https://github.com/instadeepai/bioclip) | PLM + protein structure graph; `bio_clip/model/gnn.py` | README badge states CC BY-NC-SA 4.0; root LICENSE absent/unresolved |
| 6 | [duolinwang/S-PLM](https://github.com/duolinwang/S-PLM) | sequence+structure multi-view; `model.py` | MIT |
| 7 | [hadi-abdine/Prot2Text](https://github.com/hadi-abdine/Prot2Text) | seq/structure + function text; `prot2text_model/Model.py` | CC BY-NC-SA 4.0 |
| 8 | [ColinFX/Prot2Text-V2](https://github.com/ColinFX/Prot2Text-V2) | sequence+language, contrastive stage; `scripts/train_contrast.py` | MIT |
| 9 | [mheinzinger/ProstT5](https://github.com/mheinzinger/ProstT5) | amino acid ↔ Foldseek 3Di; `scripts/translate.py` | MIT |
| 10 | [westlake-repl/SaProt](https://github.com/westlake-repl/SaProt) | sequence + structure-aware vocabulary; `model/saprot/base.py` | LICENSE present, exact SPDX not verified |
| 11 | [DFrolova/MULAN](https://github.com/DFrolova/MULAN) | sequence + angle structure; `mulan/model.py` | LICENSE present, exact SPDX not verified |
| 12 | [Emad-COMBINE-lab/clasp](https://github.com/Emad-COMBINE-lab/clasp) | sequence/structure/text tri-modal; `src/train_clasp.py` | GPL-3.0 |
| 13 | [OpenProteinAI/PoET-2](https://github.com/OpenProteinAI/PoET-2) | sequence/MSA/structure context; `src/poet_2/models/poet_2.py` | Apache-2.0 code; separate noncommercial model-weight terms |
| 14 | [bytedance/dplm](https://github.com/bytedance/dplm) | protein diffusion/structure variants; `generate_dplm2.py` | LICENSE present, exact SPDX not verified |
| 15 | [DeepGraphLearning/GearNet](https://github.com/DeepGraphLearning/GearNet) | residue graphs/structural contrastive views; `gearnet/model.py` | LICENSE present, exact SPDX not verified |
| 16 | [facebookresearch/esm](https://github.com/facebookresearch/esm) | sequence/structure model implementation; `esm/esmfold/v1/esmfold.py` | LICENSE present |
| 17 | [chao1224/ProteinDT](https://github.com/chao1224/ProteinDT) | sequence+text CLAP; `ProteinDT/models/model_ProteinText.py` | LICENSE present |
| 18 | [TencentAILabHealthcare/scPROTEIN](https://github.com/TencentAILabHealthcare/scPROTEIN) | sc-proteomics graph contrastive; `model.py` | LICENSE present |
| 19 | [mengerj/mmcontext](https://github.com/mengerj/mmcontext) | single-cell gene expression + text; `src/mmcontext/training.py` | LICENSE present |
| 20 | [MIILab-MTU/CLCLSA](https://github.com/MIILab-MTU/CLCLSA) | multi-omics views; `networks/models/clcl.py` | root LICENSE not found |

## 正反合 / 架构影响

**正题：** ProtST/CLEF/ProTrek/ProtCLIP/CLASP/ProteinDT provide code for different sequence–text/structure/feature alignments; BioCLIP/S-PLM/GearNet are direct sequence–structure/graph analogues.

**反题：** Some are not contrastive, some model antibody/natural-language/design or single-cell data, and license scope may differ between code and weights. None of these repositories by its existence proves cross-modal representations predict target-species vaccine protection.

**合题 / implementation:** Retain independent modality encoders and contrastive losses only for biologically paired inputs. Use separate heads for function, interaction, antigen recognition and protection; missing modalities use observation masks. VaccineGPT's M3 and AEP evidence grades do not promote a representation cluster into protection evidence.

**Falsifiable test:** Compare sequence-only vs sequence+paired modality under the same E1/E2 target endpoint, identity-cluster heldout split, and modality ablation; reject the multimodal increment if the pre-registered heldout metric does not improve with uncertainty bounds.
