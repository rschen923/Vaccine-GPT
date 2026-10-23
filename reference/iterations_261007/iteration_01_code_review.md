# GCDTO 迭代 01 — 蛋白质语言模型公开代码核验

**已检查 20/20 公共 GitHub 仓库。** 其中包含基础模型实现、embedding 工具、下游任务应用和教程；它们不是 20 个互不相同的基础模型架构。每个条目检查了 README、相关源码路径和 license 文件/状态。代码实现不是疫苗保护证据。

| Repository | 相关实现路径 | License 状态 / 使用边界 |
|---|---|---|
| [facebookresearch/esm](https://github.com/facebookresearch/esm) | `esm/model/esm2.py`, `esm/pretrained.py` | MIT；代码与权重/数据条款仍需分开核验 |
| [Biohub/esm](https://github.com/Biohub/esm) | `esm/models/esm3.py`, `esm/pretrained.py` | MIT；各 ESM 权重/API 的访问条件单独核验 |
| [agemagician/ProtTrans](https://github.com/agemagician/ProtTrans) | `Embedding/prott5_embedder.py` | MIT |
| [salesforce/progen](https://github.com/salesforce/progen) | `progen2/sample.py` | BSD-3-Clause |
| [huggingface/transformers](https://github.com/huggingface/transformers) | `src/transformers/models/esm/modeling_esm.py`, ESM config/tokenizer | Apache-2.0 |
| [agemagician/Ankh](https://github.com/agemagician/Ankh) | `src/ankh/extract.py` | README 指出预训练模型为 CC BY-NC-SA 4.0；源码与模型条款分开 |
| [mheinzinger/ProstT5](https://github.com/mheinzinger/ProstT5) | `scripts/embed.py`, `scripts/translate.py` | MIT |
| [nadavbra/protein_bert](https://github.com/nadavbra/protein_bert) | `proteinbert/conv_and_global_attention_model.py` | README 声明 MIT；未在根目录确认许可证文件，条款待核验 |
| [ntranoslab/esm-variants](https://github.com/ntranoslab/esm-variants) | `esm_score_missense_mutations.py` | MIT |
| [uci-cbcl/esm-efficient](https://github.com/uci-cbcl/esm-efficient) | `esme/esm.py` | MIT |
| [LucaOne/LucaOne](https://github.com/LucaOne/LucaOne) | `src/get_embedding.py` | Apache-2.0 |
| [OpenProteinAI/PoET](https://github.com/OpenProteinAI/PoET) | `scripts/score.py` | 源码 MIT；README 中权重为 CC BY-NC-SA 4.0 academic-use 条件 |
| [THUDM/ProteinLM](https://github.com/THUDM/ProteinLM) | `pretrain/pretrain_tape.py` | Apache-2.0 |
| [zjunlp/OntoProtein](https://github.com/zjunlp/OntoProtein) | `run_pretrain.py` | MIT |
| [oxpig/AbLang](https://github.com/oxpig/AbLang) | `ablang/embedding.py`, `ablang/pretrained.py` | BSD-3-Clause |
| [ai4protein/Pro-Prime](https://github.com/ai4protein/Pro-Prime) | `script/finetune.py` | LICENSE 文件为 CC BY-NC-ND 4.0；README 出现不一致措辞，须按许可证文件复核 |
| [rs239/abmap](https://github.com/rs239/abmap) | `abmap/plm_embed.py` | MIT |
| [prescient-design/lobster](https://github.com/prescient-design/lobster) | `src/lobster/model/_mlm.py` | Apache-2.0 |
| [kren-ai-lab/sylphy](https://github.com/kren-ai-lab/sylphy) | `sylphy/embedding_extractor/esm_embedding.py` | MIT |
| [Sciwhylab/protein-embedding-and-classification](https://github.com/Sciwhylab/protein-embedding-and-classification) | `tutorial_code.ipynb` | 根目录 LICENSE 可见，但获取文本失败；许可类型未核实 |

## 正反合

**正题：** ESM / ProtT5 / Ankh / AbLang 等代码可输出 frozen embeddings；PoET/ProGen 等提供生成/score 类工作流；某些工具有 residue- 与 sequence-level representations。源码核验支持“可调用的软件管线存在”这一事实。

**反题：** foundation-model representation、variant score、function/structure/antibody任务的代码存在，不代表其 output 是 VaccineGPT H1/H2/H3/H4 标签，更不证明保护效力；权重许可可能比代码许可更窄。

**合题 / 项目影响：** VaccineGPT 将 ESM-2 设为可选冻结 v1 特征后端；ESM-3 只作 comparator，不作 classifier；支持模型 checkpoint hash 和源版本记录。选型必须由同一 cluster-heldout target-task benchmark 决定，而非按仓库名称或模型规模直接认定更好。

**可证伪检测器：** 在相同 E1/E2 标签、同一 MMseqs2-family held-out split 上对比预注册的 ESM-2 与轻量表征基线；若 top-k protection-related experimental hit rate 无预设提升则拒绝 ESM-2 增量主张。

