# GCDTO 迭代 03 — 对比/多模态表征与任务迁移

**问题：** 对比学习和多模态融合在哪些 biological endpoints 有直接证据？这些结果可否迁移到疫苗保护？

**配额状态：** round 03 下载 200/200 个 OA fullTextXML。AI 子代理辅助审阅随机抽样 10 篇的摘要和所提供的 Methods/Results/Discussion 片段；不少正文节选被截断，以下只陈述可见证据，并非人工逐篇通读。完整本轮引用集在 `..\data\research_campaign_261007\rounds\round_03_references.jsonl`。

## AI 辅助抽样证据（10 篇）

| Work | 摘录中观察到的结果与位置 | 终点及边界 |
|---|---|---|
| CLAPE, DOI `10.1093/bib/bbad488`, PMC10782905 | Abstract 报告蛋白-DNA结合位点残基预测 AUROC 0.871/0.881；Results 用 ProtBert、1D-CNN、contrastive loss | 分子结合位点 ≠ 保护；结果/方法片段截断，AUROC 不等于高 precision |
| MIF2GO, DOI `10.1038/s42003-024-07411-y`, PMC11681170 | Results 报告 GO function 多模态对比预训练；MF/BP/CC M-AUPR 0.624/0.317/0.528 | GO annotation ≠ immunogenicity/protection；多个章节节选截断 |
| MultiRepPI, DOI `10.1186/s12870-025-06878-z`, PMC12275386 | 使用序列/结构/无序信息融合预测植物 peptide-protein interaction；negative 来源为随机生成 | benchmark negative 规则可能影响结果；植物 PPI ≠ 细菌抗原 |
| 肺结节 radiomics+serum, DOI `10.21037/jtd-2025-1214`, PMC12688578 | 638 个结节、633 人；验证 AUC 0.926 | 非蛋白/免疫任务，说明多模态融合可适用于其定义的诊断终点，不能外推为疫苗证据 |
| fetal-lung fusion, DOI `10.3389/fendo.2026.1832468`, PMC13223097 | 动物与人 ultrasound 任务；摘要报告动物和人群 AUC | 组织影像任务；方法/结果片段截断，不用于蛋白任务主张 |
| coRBP map, DOI `10.1038/s41467-026-71511-y`, PMC13284236 | APEX+OOPS 定位 14 个 compartment 的 1,768 个已知/候选 RBP | 共定位不是 direct physical interaction；多节选截断 |
| FlowDock, DOI `10.1016/j.apsb.2026.04.009`, PMC13366307 | Results 报告 docking pose success：CASF16 89%，PoseBusters 67%；全物理有效率 51.2% | docking/affinity ≠ in vivo protection；未完整审阅全套方法/对比 |
| M2-PRNet, DOI `10.1093/bioinformatics/btag614`, PMC13516915 | Cross-scale contrastive alignment 用于蛋白-RNA affinity；PRA310/PRA201 RMSE 1.368/1.301、PCC 0.601/0.598；按 sequence cluster 划分 folds | Interaction affinity ≠ function/protection；benchmark 有限，若要迁移需独立 endpoint |
| Genolator, DOI `10.1186/s13059-026-04274-w`, PMC13579901 | 多模态嵌入 + GO 派生 QA；>365,000 QA pairs；K-means 相似度拆分 | QA/GO 与 embedding clusters 不等于 protection 或 causal mechanism；部分结果节选无数值 |
| Top-down histone proteoforms, DOI `10.1016/j.mcpro.2026.101640`, PMC13595144 | MS fragmentation multimodal analysis；摘要报告序列覆盖 >92% | 是 proteomics measurement workflow，不是 contrastive predictor，也非疫苗终点 |

## 正题与反题

**正题：** MIF2GO、M2-PRNet、CLAPE 给出不同的 modality pairing 与 contrastive objective 能服务各自预设的 GO、亲和力、binding-residue 任务的实例证据。它们支持“配对已定义、目标特异的正样本对；评价与训练标签一致”。

**反题：** 所有样本的目标终点都不是 in-vivo vaccine protection。MultiRepPI 使用随机 negatives；CLAPE 摘要中的 AUROC 不足以说明 precision@threshold；多处 Results/Methods 片段截断。不能据此把 multimodal embedding、PPI、binding 或 GO 直接标作保护证据。

## 合题 / 架构增量

保留独立的公共/私有对比轨与 M2 分任务头，但每个 head 必须携带 endpoint 和标签 provenance；embedding 相似/聚类仅是 representation artifact，Interaction、function、immunogenicity、protection 分开存储，禁止跨终点自动晋级。增加 family-clustered validation、modality/contrastive ablation、negative-source audit。此处迁移类型为**未验证 within-field extension**，不是已验证 protection transfer。

## 可证伪假设

在含明确 challenge-protection 标签的独立数据集上，以 antigen-family cluster 作留出测试；比较同一架构的 single-modality 与 multimodal-contrastive 版本。只有 fused 模型相对最强单模态的 AUROC paired 95% group-bootstrap 差值置信区间完全大于 0，才支持对保护终点的增量价值；否则撤回该增量主张。
