# GCDTO 迭代 09 — Pareto / multiobjective / active learning

**配额：** round 09 已下载 200/200 OA fullTextXML；AI 子代理辅助审阅 10 篇摘要与供应的 Methods/Results/Discussion passages。片段被截断时只记录可见结果；不把别的 endpoint 的优化成绩迁移为 vaccine protection。这不是人工逐篇通读。

## 抽样证据

| DOI / PMCID | 正文抽样所支持 | 限制 |
|---|---|---|
| `10.3389/fgene.2025.1523015` / PMC12018315 | 1974 compounds QSAR+ADMET，PSO objectives；报告 R² 0.743 等 | excerpt 未说明 Pareto/constraints；test 50 compounds，作者仍要求lab validation；metabolizable不是无条件有利 |
| `10.3389/fimmu.2025.1586271` / PMC12130045 | computational HSV-2 multiepitope mRNA constructs，in silico antigenicity/MHC/immune simulation | 摘要要求实验/临床前 validation；非实测保护 |
| `10.1093/bib/bbaf434` / PMC12392411 (ESM2_AMP) | protein-embedding PPI/residue prediction；报告 5-fold Pan AUC>0.99、acc>0.97，另有sequence-filtered test | balanced curated benchmark ≠ real prevalence；attention/annotation overlap不是因果binding证据 |
| `10.3390/vaccines14010077` / PMC12846515 | FPV VP2 baculovirus/VLP platform，报告表达和yield、VLP结构及小动物challenge outcomes | Discussion仅每组3只猫，统计与推广有限；不是Pareto/active-learning method |
| `10.3389/fimmu.2025.1707183` / PMC12869314 | review概述 lung-cancer vaccine administration/route approaches | 二级综述；excerpt无search/evidence grading与结果效应量，非Pareto证据 |
| `10.3389/fonc.2026.1766201` / PMC12992067 | review概述 tumor antigen / immunoinformatics / design | 二级综述，不是量化模型验证 |
| `10.52225/narra.v6i1.2897` / PMC13170919 | 结核 multiepitope in silico design；摘要称 population coverage 98.19% | 结论要求in vivo；computational coverage/docking/simulation不是效果 |
| `10.1126/sciadv.aed2483` / PMC13215168 | inhalable NHS hydrogel 材料的动物/保护主张出现在摘要与Discussion | supplied Results片段未覆盖效力数据；reactivity 与 host-protein capture trade-off |
| `10.1016/j.fochx.2026.104263` / PMC13487595 | model wine protein isolate purity/recovery与白/红葡萄酒 PA removal分属多目标结果 | matrix依赖；非 Pareto-front方法，也非疫苗数据 |
| `10.1093/bioadv/vbag233` / PMC13581264 (MiLaSol) | protein solubility model报告 DeepSol ACC 81%, MCC 0.63，另基准有 recall/AUC tradeoff；模拟退火优化可溶性变体 | 作者说实验增溶数据不足，用代理构造，其他预测器确认不是实验确认 |

## 正题/反题

**正题：** 多目标优化可显露 potency/yield/toxicity/coverage 等权衡；Bayesian/active learning 可按 assay budget 选择实验。现有样本包括不同终点的计算和实验例证。

**反题：** “优化分数”不等于 protected phenotype；特定药物/葡萄酒/蛋白溶解度目标不构成 VaccineGPT objective。抽样未找到足够证据证明 Pareto front 是生物学唯一决策或多数 vaccine pipeline 已有独立前瞻 validation。

## 合题 / 决策核

保留 Pareto front 并把 `U` 仅用于前沿层内排序；安全硬门与未知输入先处理，目标效能/保护、广度、制造/免疫风险必须各自带证据级与不确定性。Pareto输出的是候选集合，不是唯一 winner；选择标准由预注册应用约束/专家治理给定。Assay acquisition 建议需针对决策影响评估，不能因模型不确定性大就自动做不安全或无效实验。

## 可证伪试验

前瞻、scaffold/family-heldout 对比 uncertainty-aware multiobjective acquisition、随机采样与预注册 single-objective baseline，固定实验预算，盲法实测 potency 与安全相关endpoint。主要指标为 measured feasible Pareto hypervolume per assay budget。若不高于两个对照、或 Pareto membership 对 assay replicate variation 极不稳定，撤回 active-learning 增益主张。
