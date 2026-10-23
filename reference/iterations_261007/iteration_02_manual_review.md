# GCDTO 迭代 02 — DNA / genome foundation models

**问题：** DNA / genome-level embedding 哪些任务有证据，可否外推到细菌 gene-level label？

**配额状态：** round 02 已下载 200/200 OA fullTextXML；下列为 AI 子代理辅助核查的随机抽取 10 篇摘要与 Methods/Results/Discussion 片段，并非人工逐篇通读或审读 200 篇全文。完整本轮参考清单在 `..\data\research_campaign_261007\rounds\round_02_references.jsonl`。

## AI 辅助抽样证据（10 篇）

| Work | 正文所报证据 | 外推限制 |
|---|---|---|
| Lamprotula genome survey, DOI `10.1093/dnares/dsaf020`, PMC12454935 | Discussion 报告 short-read assembly 平均约比 k-mer genome-size estimate 大 22%，归因复杂/杂合 genome | genome-resource 研究不是 DNA-model 预测 benchmark；强调 assembly label/error 本身可影响训练 |
| DNABERT2-CAMP promoter recognition, DOI `10.3390/genes17010027`, PMC12841110 | Abstract/Methods/Results 报 5-fold accuracy 93.10%、ROC AUC 97.28%；external accuracy 89.83%、AUC 92.79% | 8,720 条 *E. coli* 81-bp promoter/non-promoter；sigma factor 合并、非σ70样本稀少，external negative 组成节选未完全说明 |
| Onthophagus sex-specific assemblies, DOI `10.1093/gbe/evag023`, PMC12902608 | Abstract 报每个 sex-specific assembly ≥15,403 gene models、BUSCO completeness 高 | 单物种参考资源，不证明细菌基因功能迁移 |
| Virus-host prediction, DOI `10.1093/ve/veag009`, PMC13061156 | 58,046 viral genomes、15 families；k-mer NN 预测人类宿主兼容性，报告留病毒家族测试 | viral whole-genome binary host label 与 bacterial gene label 非同一任务；作者提及采样偏差、host-label 简化及低复杂序列失败 |
| Mepraia mtDNA from RNA-seq, DOI `10.1007/s00438-026-02434-y`, PMC13186837 | 18 个 partial transcript assemblies 中各恢复 13 个 mitochondrial protein-coding genes | 覆盖与表达影响 recovery；不是 gene-level language model 评估 |
| PREMIER genome editing, DOI `10.1093/nar/gkag626`, PMC13284714 | Abstract 报 targeted replacement DNA segment 可达 10.3 kb；Results 介绍 prime-editing/microhomology design | mammalian genome engineering，不是 bacterial DNA foundation model benchmark |
| Verbascum mitochondrial genome, DOI `10.1186/s12870-026-08964-2`, PMC13374249 | Results 报 478,020 bp mitochondrial assembly、36 protein-coding genes、468 RNA-editing sites | 单一植物细胞器 genome，不是预测基准 |
| LMNA variants scored by Evo 2, DOI `10.1155/humu/2444803`, PMC13508025 | Methods/Results/Discussion 描述 delta-log-likelihood 排序 50 个候选，再选 3 个做结构/dynamics 分析 | 后续是计算结构分析，不是 independent function validation；使用 Evo2 不证明细菌 gene-task 准确性 |
| Pan-cell-type epigenome-ssm, DOI `10.1186/s13059-026-04148-1`, PMC13520368 | 报告 9,539 histone tracks 压缩为 33 个 chromatin feature，用于表达/enhancer/conservation prediction | 输入为 imputed human chromatin，不是 raw DNA 或 bacterial labels |
| Dianthus chloroplast genome, DOI `10.12688/f1000research.180038.4`, PMC13562929 | Abstract/Methods/Results 报 149,567 bp plastome 并提出 variable marker loci | 单植物 plastome/phylogeny；未提供 Discussion excerpt，无 predictive DNA-model evidence |

## 正题

DNABERT2-CAMP 是序列预训练表示用于细菌启动子分类的一个近邻例证；
其它基因组模型显示序列/上下文可能有预测信号。可将这作为 “v2 DNA branch
值得检验” 的理由，但不作为架构已验证成功的结论。

## 反题

1. Promoter classification ≠ gene function、essentiality 或 protection；域、长度、标签和拆分策略均不同。
2. DNA 窗口训练会受到 assembly/annotation 质量影响；跨物种或跨 assay 迁移需要独立 held-out validation。
3. Evo2 使用于 LMNA variant prioritization 是计算优先级例子，不等于 validated bacterial gene-level accuracy。

## 合题 / 架构增量

保留 v1 protein-only 与 v2 DNA-window 两条版本边界；v2 暂用明确的 DNA window projector 与 element-type schema，未知区间输出 `unclassified_window`；ESM/Evo2 模型输出不自动进入 AEP 证据级 E1/E2。

## 可证伪假设

在 curated bacterial gene-level labels 上对比 Evo2/pretrained encoder、k-mer、CNN、随机初始化 control；按 species 与 ortholog/near-duplicate cluster 留出。预注册 success criterion 例如 held-out species macro-MCC 至少提高 0.03 且 species-level bootstrap 95% CI 高于 0。若优势只出现在随机拆分或随机初始化 baseline 同样达到，则否定 transfer 主张。
