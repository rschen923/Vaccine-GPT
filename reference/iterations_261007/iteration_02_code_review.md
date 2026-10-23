# GCDTO 迭代 02 — DNA / genome foundation code 核验

**已直接检查 20/20 公共 GitHub 仓库/项目。** 每项核验 canonical URL、README、相关 implementation/example 路径和 license-file 状态。清单包含 foundation model、embedding 工具、benchmark 和一个邻接 predictor，已注明非模型项；variant/generation families 不重复算多个仓库。代码存在不证明 VaccineGPT 生物学效果。

| # | Repository | 已检查实现路径 | License 文件/状态 |
|---:|---|---|---|
| 1 | [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) | `evo2/models.py` | Apache-2.0；Linux official / WSL2 limited; CUDA 12.1+, cuDNN 9.3+ |
| 2 | [evo-design/evo](https://github.com/evo-design/evo) | `evo/models.py` | Apache-2.0 |
| 3 | [MAGICS-LAB/DNABERT_2](https://github.com/MAGICS-LAB/DNABERT_2) | `finetune/train.py` | Apache-2.0 |
| 4 | [jerryji1993/DNABERT](https://github.com/jerryji1993/DNABERT) | `examples/run_pretrain.py` | Apache-2.0 |
| 5 | [instadeepai/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer) | `notebooks/nucleotide_transformer/inference.ipynb` | CC BY-NC-SA 4.0; verify non-commercial terms |
| 6 | [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna) | `standalone_hyenadna.py` | Apache-2.0 |
| 7 | [AIRI-Institute/GENA_LM](https://github.com/AIRI-Institute/GENA_LM) | `downstream_tasks/APARENT/run_APARENT_finetuning.py` | MIT |
| 8 | [kuleshov-group/caduceus](https://github.com/kuleshov-group/caduceus) | `caduceus/modeling_caduceus.py` | Apache-2.0 |
| 9 | [songlab-cal/gpn](https://github.com/songlab-cal/gpn) | `src/gpn/embeddings.py` | MIT; GPN/GPN-MSA/PhyloGPN/GPN-Star counted as one repository |
| 10 | [MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S) | `train/pretrain/DNABERT2_MIX/bert_layers.py` | No LICENSE/LICENCE found in inspected tree |
| 11 | [lingxusb/megaDNA](https://github.com/lingxusb/megaDNA) | `megaDNA/megadna.py` | No LICENSE/LICENCE found in inspected tree |
| 12 | [nbrg-ppcu/prokbert](https://github.com/nbrg-ppcu/prokbert) | `src/prokbert/ProkBERTDataCollator.py`, embedding example | MIT |
| 13 | [TencentAILabHealthcare/DNAGPT](https://github.com/TencentAILabHealthcare/DNAGPT) | `dna_gpt/model/dna_gpt.py` | PolyForm Noncommercial 1.0.0 |
| 14 | [DennisGankin/species-aware-DNA-LM](https://github.com/DennisGankin/species-aware-DNA-LM) | `src/datamodules/sequence_encoders.py` | No LICENSE/LICENCE found in inspected tree |
| 15 | [LucaOne/LucaOneApp](https://github.com/LucaOne/LucaOneApp) | `algorithms/batch_converter.py` | Apache-2.0; broad biological sequence scope |
| 16 | [jgi-genomeocean/genomeocean](https://github.com/jgi-genomeocean/genomeocean) | `genomeocean/embedding.py`, embedding example | Custom license agreement |
| 17 | [ramanathanlab/genslm](https://github.com/ramanathanlab/genslm) | `examples/embedding.ipynb` | MIT; example focus includes SARS-CoV-2 |
| 18 | [macwiatrak/Bacformer](https://github.com/macwiatrak/Bacformer) | `bacformer/modeling/modeling_pretraining.py` | Apache-2.0 |
| 19 | [ZjGaothu/EpiGePT](https://github.com/ZjGaothu/EpiGePT) | `model/EpiGePT.py` | MIT; adjacent task-specific chromatin predictor |
| 20 | [frederikkemarin/BEND](https://github.com/frederikkemarin/BEND) | `bend/utils/embedders.py` | BSD-3-Clause; embedding benchmark, not model |

## 正反合 / 架构影响

**正题：** Multiple public DNA/nucleotide systems implement long-context modeling, single-nucleotide sequence embeddings, microbial genomics representation, and embedding extraction.

**反题：** Evo2/GenomeOcean/viral or chromatin systems do not thereby validate bacterial gene-function or vaccine targets; model/code/weight licenses may differ. This Windows environment lacks CUDA-enabled PyTorch and the upstream Evo2 runtime.

**合题 / implementation:** Keep v1 protein and v2 DNA track strictly separated; DNA windows/reserved element schema remain v2 only. Evo2 lazy adapter is gated behind supported WSL2/CUDA, exact checkpoint hash, and target-genome validation. A DNA embedding is never presented as a protein antigen probability.

**Falsifiable test:** Compare pretrained DNA embedding with k-mer/CNN baselines on explicit bacterial gene-element labels, held out by genome and MMseqs2 90%-identity family groups. No prespecified gain means no v2 promotion.
