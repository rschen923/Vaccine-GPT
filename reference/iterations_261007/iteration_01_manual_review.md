# GCDTO 迭代 01 — 蛋白质基础模型与 ESM

**问题：** 蛋白质语言模型的何种证据可支持 VaccineGPT v1 表征；哪些性能不能外推为抗原保护？

**配额状态：** round 01 的 Europe PMC OA fullTextXML 下载配额为 200/200。随机抽取 10 篇的摘要及选定 Methods/Results/Discussion 段落由 AI 子代理辅助审阅；这 10 篇是节选级抽样，不等于人工逐篇深读或手读全部 200 篇。所有条目、DOI/PMCID、license、section index 与全文哈希见
`..\data\research_campaign_261007\rounds\round_01_references.jsonl` 与总 ledger。

## AI 辅助抽样证据（10 篇）

| Work | 本轮读取到的主张与位置 | 迁移边界 |
|---|---|---|
| NetGO 3.0, DOI `10.1016/j.gpb.2023.04.001`, PMCID `PMC10626176` | Results “Performance comparison…”：作者报告 ESM-1b embedding 的 LR-ESM 在三类 GO domain baseline 上的比较结果 | GO function prediction ≠ bacterial surface exposure/protection；摘要摘录无效应量，待查原表 |
| xCAPT5, DOI `10.1186/s12859-024-05725-6`, PMCID `PMC10924985` | Methods “Datasets and experiments”; Results/Conclusion：ProtT5 embedding + Siamese CNN/XGBoost 用于 PPI 分类，包含跨物种测试 | PPI ≠ 疫苗靶点；当前节选没有完整性能值、免疫终点 |
| PLMSearch, DOI `10.1038/s41467-024-46808-5`, PMCID `PMC10981738` | Methods “PLMSearch consists of three steps”; Discussion：embedding+PfamClan/learned similarity 用于远同源检索；讨论相对 MMseqs2 与结构搜索 | 同源搜索不是抗原验证；只在预先阅读的讨论片段观察到限制表述，需查完整上下文 |
| ESMtherm, DOI `10.1371/journal.pcbi.1012248`, PMCID `PMC11293664` | Abstract/Discussion/Conclusion：ESM-2 微调用于蛋白稳定性；对大 scaffold 性能较弱 | 研究数据集中小结构域约 40–72 aa；不能外推至表面、免疫或保护 |
| HaloClass, DOI `10.1007/s10930-024-10236-7`, PMCID `PMC11543744` | Results “HaloClass is an SVM classifier…”：278 条盐耐受数据上的 ESM-2 embedding + SVM 报告 ACC 94%、AUROC 0.99、MCC 0.88 | 盐耐受分类不是疫苗目标；新数据集测试节选被截断，结果待复核 |
| NetStart 2.0, DOI `10.1186/s12859-025-06220-2`, PMCID `PMC12366053` | Results/Discussion：ESM-2 representation 与局部上下文用于真核翻译起始位点 | 任务为真核核酸上下文，不验证细菌蛋白模型 |
| EcoFoldDB, DOI `10.1111/1462-2920.70178`, PMCID `PMC12441223` | Results/Discussion：Foldseek/ProstT5 结构导向微生物功能 profile；数据库摘要称 842 个结构关联 637 个基因 | 生态功能资源不证明疫苗靶点；摘要中灵敏度/精度结论的正文位置仍待查 |
| DeepFold-PLM, DOI `10.1093/bioinformatics/btaf579`, PMCID `PMC12598648` | Results 3.1.2：56 CASP15 monomer 上 plmMSA 报告平均 8 s，JackHMMER 365 s，约 47 倍速度；正文也记载 noisy alignment 失败 | MSA/结构效率不等于抗原功能 |
| MoCETSE, DOI `10.1371/journal.pcbi.1013397`, PMCID `PMC12998954` | “Comparison of different foundation methods”; Discussion：ESM-1b 加目标特定处理和分泌系统 transformer 用于革兰阴性菌效应蛋白 | 与细菌生物学相近但效应蛋白分类不等于免疫原性/保护；节选没有完整数值表 |
| Viral genomic PLM, DOI `10.1093/gigascience/giag081`, PMCID `PMC13495794` | Discussion 明确：attention 得到的蛋白关联是“computational priors rather than direct evidence of physical PPIs” | 病毒对象非细菌；将模型链接当实测相互作用不成立 |

## 正题

蛋白 embedding 在功能分类、远同源检索、结构/稳定性任务中能携带对下游任务有用的序列信号；MoCETSE 等工作也给出细菌效应蛋白方向的特定任务例子。

## 反题

1. 这些终点多为 GO、同源、PPI、稳定性、盐耐受或分泌，**不**是保护效力。
2. 域迁移并不保证：ESMtherm 指出大 scaffold 表现弱；DeepFold-PLM 记录 alignment 噪声；交互关联可能只是模型先验。

## 合题 / 架构增量

保留冻结 ESM-2 + 任务特定小模块作为 v1 候选实现，但把 PLM 输出严格视作 feature，不允许其单独作靶点裁决；将 ESM-3 留作 comparator，而不是疫苗分类器。M2/AEP 必须用目标菌独立标签、实验读出和证据级共同决策。

## 可证伪预测

预注册 leave-one-species-out 与同源簇隔离比较：传统细菌候选基线 vs 加 PLM 表征模型，盲法验证 top-k 的表面暴露与抗原识别。若 top-k 验证率提升 <10 percentage points，或置信区间包含零，则拒绝“PLM 带来有用增量排序价值”的主张。该试验不直接证明保护效力。
