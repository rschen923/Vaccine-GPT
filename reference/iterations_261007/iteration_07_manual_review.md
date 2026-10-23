# GCDTO 迭代 07 — Biological graph / attention claims

**配额：** round 07 下载 200/200 OA fullTextXML；AI 子代理辅助审阅随机 packet 10 篇的摘要和已提供 Methods/Results/Discussion 摘录；节选标截断者没有向下推断。这不是人工逐篇通读。完整参考集：`..\data\research_campaign_261007\rounds\round_07_references.jsonl`。

## 抽样证据（10 篇）

| DOI / PMCID | 抽样段落直接支持的内容 | 迁移限制 |
|---|---|---|
| PIQLE, `10.1093/bioadv/vbad070`, PMC10281963 | 2 Methods、2.5 Datasets、2.6 Metrics、3 Results、4 Conclusion：multi-head graph attention 评分 protein-complex interface quality；报告 Dockground ρ=0.519、HAF2 ρ=0.429 | finite decoy-set DockQ 排序，不是因果 residue interaction；22个targets量级有限，Results节选截断 |
| “Experimental analysis of graph representation learning for GO protein-function prediction”, `10.7717/peerj.18509`, PMC11569786 | survey 的 GO/PPI/structure graph-learning 方法族；结论列 noisy/incomplete PPI、annotation sparsity、oversmoothing | 二级综述非统一 controlled comparison；GNN attention不等于 biological causality |
| DrugMAN, `10.1186/s12859-024-05976-3`, PMC11577831 | Methods/Results 描述多药物/蛋白网络与 mutual attention DTI；数据集20,565 binding pairs，随机 unseen-pair negatives，7:1:2 split | randomly-unobserved pairs 未必实验 negative；缺失标签与网络重用会影响泛化 |
| GTPLM-GO, `10.3390/ijms26094088`, PMC12072039 | Results/Methods 报 PPI＋sequence features 做 GO prediction；相对 DeepGraphGO 报 Fmax/AUPR 提升并用CAFA式 timestamp split | 训练样本可用性不同、STRING 可能含未来信息；GO预测不证明 graph causal effect |
| protein structure graph for missense pathogenicity, `10.1093/nargab/lqaf097`, PMC12288871 | Methods/Results 描述 ESMFold graph、protein-family-filtered external test；所提供 Results 片段显示 graph-only 不及 CADD，mutated-residue embedding 比 pooled好 | 摘要声称增强 CADD 的结论未在 supplied result 段得到完整量化；variant effect ≠ GNN edge causality |
| MGMA-PPIS, `10.1093/gigascience/giaf114`, PMC12486388 | multiscale EGNN/EGAT 预测 residue PPI site；5-fold 与独立测试集 | 需结构，数据 reused benchmark；若未按 sequence identity cluster split 仍可能同源泄漏 |
| DCGAT-DTI, `10.1093/bioadv/vbaf306`, PMC12776360 | abstract/results 声称动态图跨模态邻居提升 warm/cold DTI；embedding相似度构图 | Results 数值节选不充分，阈值敏感，predicted similarity edge 不是实测 binding |
| AI methods for understudied proteins, `10.1186/s13040-026-00577-7`, PMC13523371 | review 讨论 disorder/flexibility、hallucination、apo/holo 和需 mutagenesis/assay 验证 | 二级综述、异质引用；模型复合物/静态结构预测不能独自验证功能 |
| GNNs for gene–drug modeling in cancer, `10.3389/fonc.2026.1929697`, PMC13550853 | review 指出若干 gene/drug target nominations 是 computational、未在对应 cancer model 实验验证 | cancer 任务与 pathogen graph 不同；文献 prior/attention edge 不证明 intervention causality |
| Graphene, `10.1016/j.patter.2022.100651`, PMC9868676 | self-supervised 多网络 integration 做 pathway/disease-gene reprioritization；图由多个知识网络合并 | prior curated knowledge 可造成 circularity/source leakage；attention ranking ≠ causal gene–disease relation |

## 正题与反题

**正题：** GATv2/PyG/DGL 等提供动态邻居加权实现；PPI/structure/GRN 图任务证明在各自数据和 endpoint 上可运行；GraphReg/Graphene 类工作体现图结构/多视图信息有时可增加预测信号。

**反题：** attention weights 是给定图内的 message aggregation coefficient，不是干预效应或分子直接作用证据。任务端点从 DockQ、GO、PPI site、DTI 到 cancer predictor 各不相同，不能视作疫苗靶点排名的直接验证。样本同源、随机 negatives、静态 STRING 和 source leakage 都可能让测试分数高估。

## 合题 / 架构增量

维持 GCE 的 evidence-typed graph，但严格将图编码输出称为 contextual representation/预测特征；不得以 attention 自身作机制解释。PPI/STRING边保持 E3，目标菌的测量边才可升级 E2。未验证 operon graph-GNN transfer 的单独实现；operon adjacency 当前只是可审计图边先验，不宣称学习到转录因果。

## 可证伪检测器

以 genome/protein-family heldout 划分比较 no-GCE、完整图、degree-preserving rewired graph 与各 edge-type ablation；同一参数预算、相同训练/标签。若 rewired 图保持同样表现、或 group-heldout 下增益消失，则拒绝该类边/attention 的生物学解释，并报告性能而非因果。
