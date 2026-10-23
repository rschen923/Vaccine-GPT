# GCDTO 迭代 08 — Calibration / conformal / uncertainty

**配额：** 200/200 OA fullTextXML。AI 子代理辅助审阅随机抽样 10 篇的摘要与所提供的 Methods/Results/Discussion 片段；片段截断处标记为待全文复核。这不是人工逐篇通读。完整参考集合与许可证见本轮 JSONL ledger。

## 抽样工作

| Work | 抽样证据/精确位置 | 限制 |
|---|---|---|
| HCP impurity MS quantification, DOI `10.1007/s00216-023-05024-8`, PMC10761545 | Results/discussion：已知 impurity fractions 校准 MS1 intensity ratios；n=3 repeats，多类 calibrator | 特定 HbA2 assay，不是 conformal/group coverage，也不证明跨域 transfer |
| CatPred enzyme kinetics, DOI `10.1038/s41467-025-57215-9`, PMC11871309 | Dataset splitting：序列/SMILES pair 不跨 split；sequence identity cutoffs 99/80/60/40%；Discussion 报 aleatoric>epistemic | predictive uncertainty ≠ conformal interval coverage；pair separation不保证family exchangeability |
| Traction force microscopy, DOI `10.1371/journal.pcbi.1013079`, PMC12251289 | Methods/Results：bootstrap PIV uncertainty + hierarchical Bayesian propagation | 不属于 conformal；measurement-error pipeline 的实验图像域不同 |
| DEGU genomic DL uncertainty, DOI `10.1038/s44387-025-00053-3`, PMC12779563 | Abstract称 conformal 有 coverage guarantees；Methods描述 reverse-complement pair 同fold、lentiMPRA replicate mean/SD | supplied text 未给 conformal calibration procedure、split topology、coverage results，故不得当无条件保证 |
| CHARLS edentulism risk, DOI `10.1016/j.identj.2025.109352`, PMC12800401 | Methods/Results/Discussion：独立验证、logistic recalibration；AUC 0.695；讨论校准较稳定 | 在 CHARLS 内的 clinical validation ≠ 新人群/蛋白家族 exchangeability |
| Cowpea NIRS/GWAS, DOI `10.1093/g3journal/jkag088`, PMC13232527 | Methods/Results：NIRS protein equation R² 0.86；不同 seed preparation 下与 wet chemistry 的相关不同 | 说明测量/校准依赖 protocol，不是统计 conformal coverage |
| Agroecological adaptation review, DOI `10.3389/fbinf.2026.1886248`, PMC13385113 | Discussion：要求 domain 独立定义、假设显式、先做模型 adequacy/evaluation | Perspective 不是定量验证，不证明某算法覆盖 |
| CKM sarcopenia nomogram, DOI `10.3389/fpubh.2026.1887503`, PMC13437309 | Results：多个外部 cohort AUC 与 calibration drift；hospital prevalence不同 | AUC 保持不能保证概率 calibration；clinical setting不同 |
| Developmental “recalibration” perspective, DOI `10.3389/fpsyt.2026.1867979`, PMC13519805 | Limitations：观察性箭头不证明 mediation，需 prospectively measure | “calibration”是概念临床用语，不是 statistical calibration |
| T2DM cardiovascular model, DOI `10.1021/acsomega.6c05628`, PMC13520431 | Abstract 报 nested CV、AUC 0.958、Brier 0.089 及年龄 calibration imbalance；Methods/Results节选 | 内部 nested CV 不建立外部/group coverage；摘要与数据子集分母需核实 |

## 正反合

**正题：** split-conformal、概率校准、uncertainty metrics 和 FDR/FNR控制有成熟 software/研究案例；CatPred与临床 validation 研究还说明 group/distance分层或 calibration drift应显式呈现。

**反题：** 未证据显示样本间 exchangeability 时，coverage guarantee不成立。AUROC、Brier、ECE、模型不确定性、随机 split 不等于 conformal coverage。DEGU 的摘要式“minimal assumptions”不能忽略实际 calibration/test group结构。

**合题 / 架构改动：** VaccineGPT 将三件事分开：预测概率校准、conformal区间、最终排序。要声明 90% coverage，先定 family/group 单位，训练/校准/测试按 group隔离；无代表性 calibration set 时输出 unavailable/全范围，不生成候选推荐。域转移必须独立校准/验证。

## 可证伪检测器

预注册 α=0.10；以 homology/family cluster 为 conformal score 校准和拆分单位，并另做 site/time target-domain holdout。报告 family-level coverage、interval width、subgroup coverage 与 group-bootstrap CI。下置信界若低于预设90% gate，G-C fails；即使 pass，也只支持与 tested calibration/test groups exchangeable的范围。
