# VaccineGPT M2 公开监督数据续查（261007）

## 结论

找到一项**直接以 *Edwardsiella piscicida* EIB202 为实验菌株**的公开研究，可为 M2 的 H3（宿主细胞内转运/效应蛋白）提供少量直接阳性证据。已将论文 Table 1 中能与本地 EIB202 蛋白组按 locus tag 精确对应的 10 个候选整理为 JSONL；项目现有证据分类器将这 10 行判为 E1。

这仍不足以训练 M2：数据只有阳性，没有同一实验中的可核验阴性；研究共报告 25 个转位阳性候选，但当前可下载正文仅逐项列出其中 10 个，补充 PDF 下载只返回了 PMC 的 JavaScript 下载交接页；也没有为这 10 条分配 `mmseqs2_90` 分组。因此不运行 M2，也不将结果解释成疫苗抗原或保护性证据。

H2（抗原/保护）方面仍未找到足以形成 EIB202 监督集的公开数据。已发现的跨菌株 *E. piscicida* 蛋白疫苗研究只测试 GroEL、COPIII、GlyZip 三种蛋白，其保护结果不显著或呈反保护，不能补成 EIB202 的金标签。IEDB 的 *E. piscicida* 查询没有得到可用记录；宽泛的 *Edwardsiella* 查询结果中有一条 *E. tarda* 的 PstA 肽 MHC 结合阴性记录，不能迁移为 EIB202 蛋白疫苗阴性。

## 新发现：EIB202 效应蛋白直接实验

Zhang 等报告先从感染相关转录变化中选取 200 个候选基因，再使用 TEM1 β-内酰胺酶融合报告检测 EIB202 向 HeLa 细胞内的蛋白转位。论文报告 25/200 个候选达到其阳性筛选条件。研究进一步验证了多条转运路径；Table 1 明列 10 个效应蛋白及其 EIB202 locus tag。本文只把该表中明列的这 10 个已测阳性记录纳入整理集，不把其余候选或未列出的基因推作阴性。

| EIB202 locus tag | 本地基因符号 | 本地注释 | 论文列出的通路 |
|---|---|---|---|
| ETAE_0866 | eseG | hypothetical protein | T3SS |
| ETAE_0888 | 未提供 | putative TTSS effector protein | T3SS |
| ETAE_1757 | 未提供 | hypothetical protein | T3SS |
| ETAE_1586 | 未提供 | hypothetical protein | T3SS |
| ETAE_1303 | cspA | major cold shock protein | T6SS |
| ETAE_2136 | cspI | major cold shock protein | T6SS |
| ETAE_2428 | evpP | type VI secretion system protein EvpP | T6SS |
| ETAE_0247 | sodC | copper-zinc superoxide dismutase | OMV |
| ETAE_0490 | 未提供 | putative cytochrome | OMV |
| ETAE_2080 | osmB | lipoprotein | OMV |

Locus tag 与本地蛋白记录为**精确 ID 对应**，不是同源迁移。论文结果支持 H3 的“效应蛋白/转位”范围；HeLa reporter 实验并不证明其为鱼类保护性抗原，不足以标注 H2。论文中“25/200”也不是可直接导入训练器的 25 对 175 标签：其余 175 行的逐基因检测读数、表达/可测性及原始图像计数未从当前取得的正文中恢复。

## 可复核新增文件

新增文件位于仓库外数据目录（相对于仓库根目录 `..\data\`），未混入版本控制：

- `..\data\raw\public_training_sources\m2_public_candidates\PMC5820615\PMC5820615.fullTextXML.xml`：Europe PMC 获取的完整 JATS XML。
- `..\data\raw\public_training_sources\m2_public_candidates\PMC5820615\curated_h3_eib202_positive_candidates.jsonl`：10 条 EIB202 H3 阳性候选。包括论文与 assay provenance、本地 locus/protein crosswalk、序列 SHA-256、标签解释和未分配的同源簇状态。
- `..\data\raw\public_training_sources\m2_public_candidates\PMC5820615\source_manifest.json`：URL、获取状态、SHA-256、字节数及训练阻塞项。
- `..\data\raw\public_training_sources\m2_public_candidates\PMC5820615\Presentation1.download_gate.html`：PMC 对补充文件请求返回的 JavaScript 下载交接页，**不是论文补充 PDF，也未作为研究数据解析**。

整理集的摘要：10 行、10 个不同 locus、10 个 label=1、0 个 label=0；逐行调用项目 `classify_label_evidence` 均为 E1。该 E1 仅表示有目标菌株直接实验与逐行引用，不表示该集合已适合模型拟合。manifest 记录的目标蛋白组 SHA-256 用于固定 crosswalk 所依赖的本地输入版本。

## 与其他已查资源的边界

1. **IEDB**：项目另有大规模通用表位记录缓存，但表位肽、MHC 结合、免疫原性及保护效果是不同终点，且菌种/菌株、宿主与实验结果必须逐行核验。此前按 *E. piscicida* taxon 1263550 查询没有可用于此任务的直接记录；宽泛 *Edwardsiella* 查询得到的 *E. tarda* PstA 记录只涉及肽 MHC 结合。该次 API 响应快照并未保存在本轮新增目录，因此这些零/单条检索结果应视为检索记录，而不是已归档的可复现 API 数据集。
2. **蛋白疫苗论文**：PMID 41213346、PMCID PMC13542899、DOI 10.1016/j.fsi.2025.110992；研究菌株为 *E. piscicida* S11-285、宿主为 Chinook salmon。GroEL 结果为反保护；COPIII 与 GlyZip 的注射/口服结果均轻微且未显著。已保存的全文和补充材料在 `..\data\raw\public_training_sources\m2_public_candidates\PMC13542899\`。这些证据可用于候选优先级与实验设计讨论，但不可作为 EIB202 金标签，也不能把统计不显著解释为“确认无效”。

   已将补充表 S3 的三条 cDNA 翻译后与 EIB202 蛋白组逐条精确匹配。源序列在 S3 中没有注明来自哪个分离株；下表的免疫/攻毒结果来自 S11-285，不能随序列完全相同而改写成 EIB202 结果。

   | 候选 | EIB202 精确序列对应 | 氨基酸长度 | S11-285 攻毒结果（IC / PO） |
   |---|---|---:|---|
   | GroEL | ETAE_0314, `groEL`, ACY83161.1 | 547 | RPS −184% / −76%，作者描述为反保护 |
   | COPIII | ETAE_3498, `hemN2`, ACY86329.1 | 457 | RPS +6% / −15%，变化轻微且未显著 |
   | GlyZip | ETAE_1685, `pcpY`, ACY84524.1 | 155 | RPS +9.7% / +13%，变化轻微且未显著 |

   `IC` 为腔内注射，`PO` 为口服；每个挑战处理组 24 条鱼。该结果说明三种公开疫苗构建蛋白均可在 EIB202 蛋白组中找到完全相同的氨基酸序列，但效应是在 S11-285 挑战实验、Chinook salmon 宿主、原研究给药与佐剂条件下测得。它既不能支持“EIB202 已被验证有效/无效”，也不能把 GroEL 的反保护结果简单推广至其他宿主、菌株或制剂。

   新增 `..\data\raw\public_training_sources\m2_public_candidates\PMC13542899\candidate_sequence_crosswalk_eib202.jsonl` 和 `candidate_sequence_crosswalk_manifest.json`，记录翻译长度、全长 exact match、目标 locus/protein accession、结果归属菌株和 `h2_binary_label=null`。该字段为空是有意的：这是跨菌株的连续保护结果，不是 EIB202 H2 金标签。
3. **Tn-seq**：已有 EIB202 公开 FASTQ 并完成下载校验，但 FASTQ 仍未转为经过重复性、插入位点、覆盖度、适配子/引物和基因映射核验的 gene-level fitness 标签。必需性/体外 fitness 也不能替代保护性抗原标签。
4. **Protegen**：此前官方站点请求超时；没有从搜索摘要推断条目数或标签内容。

## 是否可开始 M2 监督训练

**当前不能。** 可开始保存和审阅上述 H3 阳性记录，但 `fit_supervised_heads` 还要求每个标签对齐 128 维嵌入、每基因有效分组，并按 `grouped_split` 划分。当前 H3 整理集没有阴性、没有 `mmseqs2_90` group ID，且仅覆盖论文表中 10 个阳性。只有阳性样本无法估计区分能力；强行将其他基因当负类会把“未测/不可表达/未报告”混成真实阴性。不得为满足训练入口而补造 0 标签。

H1–H4 的当前证据与最低补数要求如下：

| 任务 | 当前可用证据 | 仍缺的训练关键项 |
|---|---|---|
| H1：宿主体内定植/必需性 | EIB202 Tn-seq 原始 FASTQ；已知论文级实验不是全基因目标菌标签 | 可复核的基因级 fitness、独立生物学重复、宿主/组织/时间与感染条件、输入/输出计数、分析对照及不确定性；保留条件依赖性，不将缺测当非必需 |
| H2：抗原/保护 | S11-285 的少量蛋白疫苗研究；IEDB 有通用表位但无可用的目标菌株保护标签 | 目标菌株蛋白身份/表达与免疫可及性证据；逐候选、同宿主和同攻毒条件下的免疫/保护结果；未免疫/载体/佐剂对照、样本量、剂量途径、时间点及统计量；明确区分结合、免疫原性与保护终点 |
| H3：效应/转位 | 新增 10 条 EIB202 直接阳性 | 找回全 25 个逐基因检测记录及其 assay 可测性；从同一实验定义可核验阴性/边界样本；统一 EIB202 locus 与蛋白序列映射并补 `mmseqs2_90` 分组；多研究验证 |
| H4：体外核心必需性 | 与 H1 不同，Tn-seq 可支持条件特异生长适应度；现有 FASTQ 尚未形成金标签 | 条件定义明确的独立文库重复、基因级插入计数/fitness 与质量控制；区分必需、适应度降低、未覆盖和未测；避免用单一培养条件泛化为普遍必需 |

**建议的记录字段**：`gene_id`、`protein_id`、目标菌株/基因组 accession、序列与 crosswalk 方法、task、实验终点及方向、label、实验/对照组、宿主与病原条件、重复数、原始计数或效应量、置信区间/显著性、可测性/覆盖度、论文/数据 accession、证据定位、同源簇分组和人工复核状态。缺失、未测、无覆盖、低于检测限、明确阴性应使用不同状态；只把预先定义并实际测量的阴性编码为 label=0。

## 引用

Zhang L, Jiang Z, Fang S, et al. Systematic Identification of Intracellular-Translocated Candidate Effectors in *Edwardsiella piscicida*[J]. *Frontiers in Cellular and Infection Microbiology*, 2018, 8:37. DOI: [10.3389/fcimb.2018.00037](https://doi.org/10.3389/fcimb.2018.00037). PMID: 29503811; PMCID: PMC5820615. 正文证据位置：Methods “TEM1 protein translocation assay”；Results “Identification of *E. piscicida* effector candidates based on macrophage infection-responsive genes”；Figure 1；Table 1。

Jacobsen KL, Griffin MJ, Phinney BS, et al. Use of shotgun immunoproteomics for the development of protein vaccines against *Edwardsiella piscicida*[J]. *Fish & Shellfish Immunology*, 2025, 168:110992. DOI: [10.1016/j.fsi.2025.110992](https://doi.org/10.1016/j.fsi.2025.110992). PMID: 41213346; PMCID: PMC13542899. 正文证据位置：Abstract、protein vaccination/challenge results、Table 3；序列见 Supplementary Table 3。
