# VaccineGPT：细菌疫苗靶点智能发现的计算—实验闭环

**版本**：1.0  
**日期**：2026-09-15  
**范围**：CLEF 起点、M1 基因组智能核心（GIC）、M2 序列设计引擎（SDE）、公开数据库、Tn-seq/CRISPRi-seq/RNA-seq 实验闭环

> 本文是项目设计综述，不把尚未完成的下载、训练或实验结果写成已证实结论。所有数值若不是文献直接报告，均标为建议参数、工程假设或待本地报价。

## 摘要

细菌疫苗靶点发现同时受三个瓶颈限制：必需性和毒力是条件依赖的，免疫原性受宿主与抗原呈递背景影响，而公开数据的物种、菌株、实验条件和序列同源性会造成严重的标签噪声与评估泄漏。VaccineGPT 的合理路线不是训练一个端到端黑箱，而是以预训练序列模型作为冻结或低秩可调的表示起点，以 M1 融合蛋白、DNA、ncRNA 和组学证据，以 M2 分别预测必需性、毒力、免疫原性并进行候选排序，再用 Tn-seq、CRISPRi-seq、RNA-seq 和定向实验形成主动学习闭环。

项目当前的软件骨架已经具备 UDC-02 特征、UDC-03 标签、UDC-04 七关系图、分组切分、四级证据权重和训练接口。下一阶段的核心不是继续堆叠模型，而是把每一个数据库记录变成可审计证据，把每一个预测变成可解释的实验假设，并把失败实验反馈到数据和采样策略中。本文据此提出：①基于同源簇、基因组/操纵子和实验条件的泄漏安全评估；②以大肠杆菌和 Edwardsiella piscicida 为主线的跨物种迁移；③M1/M2 的不确定性、校准和自纠察；④三阶段实验设计与预算框架。

## 1. 项目总体分析

### 1.1 科学问题

中心科学问题是：**在菌株、培养条件和宿主免疫背景存在差异的情况下，如何从多组学证据中识别同时满足“病原生存依赖、致病相关、可被宿主免疫识别且可工程化表达”的疫苗候选，并用最少的实验获得最大信息增益？**

该问题至少包含四个不同终点：

| 终点 | 当前模块 | 推荐标签 | 主要混淆因素 |
|---|---|---|---|
| 生存必需性 | M2-T1 | Tn-seq/CRISPRi 条件化效应 | 培养基、感染阶段、极性效应、文库覆盖 |
| 毒力/宿主适应 | M2-T2 | VFDB、感染模型、转录响应 | 物种和菌株差异、表达时相 |
| 免疫可及性 | M2-T3 | IEDB/MHC-II/表面定位/抗原加工 | 宿主 MHC、表达量、构象、免疫偏倚 |
| 综合候选排序 | M2-T4 | 多任务后验与约束 | 标签缺失、任务相关性、可制造性 |

这些终点不应被强行合并为单一“好靶点”标签。T4 应是带约束的排序函数，而不是独立真值来源。

### 1.2 当前架构的优势与风险

**优势**：  

1. 预训练模型把大量序列先验迁移到小样本任务，适合目前实验数据有限的现实条件。
2. UDC 和 lineage 设计使特征、标签、模型和批次可以追溯。
3. M1 的多视图与图关系适合表达同一基因的蛋白、核酸、操纵子、通路和互作上下文。
4. M2 将分类、回归、免疫特征和排序分离，便于使用任务特定损失。
5. 证据层级和有效样本数权重比重复复制稀有记录更安全。

**主要风险**：

1. **数据库同源泄漏**：随机按基因切分会把同一蛋白家族或同一基因组的近邻同时放入训练和测试。
2. **条件错配**：一个物种的体外必需基因不能自动转化为另一个宿主环境中的必需基因。
3. **负样本伪造**：把“未在数据库中报道”为负例会把未测量误当作阴性。
4. **多任务标签不平衡**：T1/T2/T3 的缺失模式不同，简单平均损失会让数据多的任务支配训练。
5. **预训练语料污染**：公开基因组与下游测试集可能重叠，导致模型评估偏乐观。
6. **免疫原性过度代理**：MHC-II 结合或表面定位只是免疫原性的组成因素，不等同于保护性免疫。

## 2. 相关模型、数据库与代码路线

### 2.1 序列基础模型

Lin 等在 ESM-2 中展示了仅用蛋白序列语言建模即可学习多层结构和功能信号，代表论文为 *Evolutionary-scale prediction of atomic-level protein structure with a language model*（Science, 2023, DOI: [10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)）。对 VaccineGPT 的启示是：ESM-2 适合提供通用蛋白表示，但不应被当作菌株条件或宿主免疫的替代测量。

Elnaggar 等的 ProtTrans（IEEE TPAMI, 2022, DOI: [10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)）以及 ProteomeLM 路线强调大规模蛋白语料的迁移能力。工程上，ProteomeLM 可作为蛋白视图的补充，但需要显式记录模型版本、池化层和是否冻结；不同模型的向量不应未经校准直接拼接。

DNABERT 将 DNA k-mer 建模引入 BERT 框架（Ji 等，Bioinformatics, 2021, DOI: [10.1093/bioinformatics/btab083](https://doi.org/10.1093/bioinformatics/btab083)），DNABERT-2 进一步采用更高效的 tokenizer 和跨物种预训练（Zhou 等，arXiv:2306.15006，[代码](https://github.com/MAGICS-LAB/DNABERT_2)）。其对 M1 的价值主要在启动子、操纵子和调控区表示，而不是替代 RNA-seq 的表达测量。

RNA-FM 直接学习 RNA 序列表示（Lee 等，NAR Genomics and Bioinformatics, 2022, DOI: [10.1093/nargab/lqac012](https://doi.org/10.1093/nargab/lqac012)）。Evo2 是更广义的 DNA/基因组语言模型路线。两者应分别作为 RNA 与 DNA 的候选后端，不能因为接口维度相同而混称。

**模型选择结论**：默认冻结 ESM-2/DNABERT-2/RNA-FM/Evo2，训练投影层、图层和 M2 头；当每个任务拥有足够的菌株分组数据后，才逐步开放最后若干层或 LoRA。所有比较必须采用相同分组切分和相同下游头。

### 2.2 必需性与功能数据库

DEG 是早期必需基因资源（Zhang 等，Nucleic Acids Research, 2004）；OGEE、CEG、NetGenes 和 Tn-seq/CRISPRi 研究补充了物种和条件信息。数据库的共同局限是实验体系异质、基因命名不统一、阴性证据较少。因此 T1 标签应保存 `organism`、`strain`、`condition`、`assay`、`effect_size` 和 `evidence_level`，而不是只存 0/1。

Tn-seq/TraDIS 的高通量插入计数适合估计基因组范围的适应性效应；Langridge 等的 TraDIS 工作（Genome Research, 2009, DOI: [10.1101/gr.094755.109](https://doi.org/10.1101/gr.094755.109)）奠定了大规模插入测序的路线。CRISPRi-seq 则可以对难以插入或条件敏感的基因进行可调抑制，但存在脱靶、极性和 knockdown 程度不均一问题。二者应作为互补证据而不是简单平均。

### 2.3 毒力、互作与免疫资源

VFDB 提供毒力因子及其家族注释（Chen 等，NAR, 2016, DOI: [10.1093/nar/gkv1239](https://doi.org/10.1093/nar/gkv1239)）；STRING 汇总蛋白互作和功能关联（Szklarczyk 等，NAR, 2023, DOI: [10.1093/nar/gkac1000](https://doi.org/10.1093/nar/gkac1000)）；IEDB 提供表位和免疫实验记录（Vita 等，NAR, 2019, DOI: [10.1093/nar/gky1006](https://doi.org/10.1093/nar/gky1006)）。这些资源可以支持 T2/T3，但不能绕过菌株特异实验。

IEDB 的实验数据通常包含宿主、MHC、肽段、实验类型和响应信息。M2-T3 应将结合、呈递、T 细胞反应和保护性终点分层；预测得到的 MHC-II 15-mer 只能作为软标签或候选生成信号。

### 2.4 现有代码路线的优劣

上游仓库的优势在于权重和训练接口开放、社区复现路径清晰；其不足是预训练模型的任务目标与细菌疫苗候选排序并不相同。VaccineGPT 的代码补层应集中在：

- 数据源版本、校验和拒绝记录；
- 同源/基因组/操纵子级切分；
- 证据分层与缺失标签掩码；
- 多任务不确定性和校准；
- 模型—实验主动学习闭环。

不推荐将所有上游仓库复制进本项目。应通过 lazy adapter、模型 manifest 和权重哈希调用原始模型，并固定池化、tokenizer、层和冻结策略。

## 3. 推荐的计算改进

### 3.1 高效使用全部数据

1. **不复制记录**：采用 effective-number 权重、证据 tier 权重和任务掩码。
2. **半监督表示学习**：对无标签蛋白/DNA/RNA 只训练投影一致性或对比目标，不把伪标签当作 L1。
3. **多视图缺失建模**：用 view mask 和 modality dropout，避免丢弃只有蛋白或只有 DNA 的记录。
4. **条件化标签**：将培养基、温度、宿主、感染时相作为条件变量或分层评估变量。
5. **难例挖掘**：只在训练 split 内按不确定性和错误类型选择难例，防止测试泄漏。
6. **校准后排序**：先对 T1/T2/T3 分别校准，再用可解释约束组合 T4。

### 3.2 自纠察与自优化机制

项目新增 `scripts/self_check.py` 和 `shared/self_correction.py`。它们检查空数据、ID 失配、重复、任务稀疏和 split overlap，并根据审计结果给出保守训练策略。建议后续扩展为四层：

1. **数据层**：内容类型、序列字母表、长度、重复、来源和 lineage。
2. **标签层**：证据冲突、任务缺失、类别比例、跨数据库矛盾。
3. **模型层**：校准误差、分组性能、OOD 检测、梯度异常、训练—验证差距。
4. **实验层**：候选的预测置信度、互补证据、可操作性和预期信息增益。

任何 blocking 级错误都应阻止正式训练；warning 只能生成带审计记录的临时模型。

## 4. 实验设计

### 4.1 总体策略

以大肠杆菌作为方法学和质控菌株，以 Edwardsiella piscicida 作为主病原菌，以手头其他鱼类病原菌作为外部迁移集合。实验分为：

- **阶段 A：条件化必需性图谱**（Tn-seq + CRISPRi-seq + RNA-seq）
- **阶段 B：候选抗原与毒力关联验证**（基因操作、表达、定位和表型）
- **阶段 C：鱼类相关免疫与保护性验证**（抗原制备、免疫、攻毒或替代性免疫终点）

计算模型在 A 阶段完成第一次更新，在 B 阶段完成因果标签更新，在 C 阶段验证 T4 的实际排序价值。

### 4.2 阶段 A：Tn-seq/CRISPRi-seq/RNA-seq

**设计思路**：用两个独立扰动体系测量“生长必需性”和“条件适应性”，用 RNA-seq 解释机制，避免把单一 assay 的统计缺失当成生物学阴性。

**组别**：

| 菌株 | 条件 | Tn-seq | CRISPRi-seq | RNA-seq |
|---|---|---:|---:|---:|
| E. coli | 标准培养基 | 3 生物重复 | 3 生物重复 | 3 |
| E. coli | 鱼类宿主相关应激/血清模拟条件 | 3 | 3 | 3 |
| E. piscicida | 标准培养基 | 3 | 3 | 3 |
| E. piscicida | 鱼类血清或感染相关条件 | 3 | 3 | 3 |
| 其他鱼类病原菌 | 资源允许的关键条件 | 3 | 3 | 3 |

每一条件至少保留独立文库批次或独立培养批次。Tn-seq 的关键 QC 是插入位点覆盖、读段唯一比对比例、基因级覆盖和重复相关性；CRISPRi-seq 的关键 QC 是 sgRNA 分布、非靶向对照、抑制效率和极性影响；RNA-seq 的关键 QC 是 RNA 完整性、文库复杂度、比对率和批次平衡。

**分析计算**：

- Tn-seq：以基因插入计数为输入，使用 MAGenTA/TRANSIT/edgeR 类负二项模型，输出 log2 fold-change、标准误、FDR 和条件交互项。
- CRISPRi-seq：以 sgRNA 计数为输入，按基因聚合并保留 guide-level 离散度；至少加入 non-targeting 和多个独立 sgRNA。
- RNA-seq：使用 DESeq2/edgeR 设计矩阵 `~ batch + condition`，对感染/应激条件输出差异表达与通路富集。
- 综合标签：`T1 = f(effect_size, FDR, assay_replicates, condition)`，将单 assay、重复一致但条件特异和多 assay 一致分别映射到 L2/L1 等级。

**预期判据**：

1. 生物重复相关性和测序 QC 达到预先登记阈值；
2. 多 assay 对同一候选的方向一致性优先于单一显著性；
3. 条件交互项显著的基因进入“宿主环境条件靶点”子集，而非被平均掉；
4. RNA-seq 提供通路与补偿效应解释，不单独决定必需性。

### 4.3 阶段 B：候选验证

从 T4 最高分中按“高置信度—高不确定性—跨模型分歧—可操作性”四类各选 5–10 个候选，避免只验证模型最自信的同一类基因。

建议验证：

1. 独立 CRISPRi guide 或等基因回补，确认因果性；
2. 生长曲线、竞争指数、血清耐受和氧化/渗透应激；
3. qPCR 或 targeted RNA-seq 验证表达方向；
4. 亚细胞定位、表面暴露或分泌预测与实验验证；
5. 重组蛋白纯化、聚集/稳定性和保守性评估；
6. 对 E. piscicida 优先增加鱼类血清、黏液或巨噬细胞相关条件。

### 4.4 阶段 C：免疫和保护性实验

在符合动物伦理和生物安全审批的前提下，按鱼种、菌株和抗原制定随机化、盲法和预先终点。最小设计是候选抗原组、佐剂对照、阴性蛋白对照、未免疫攻毒对照和阳性疫苗对照。主要终点可包括相对存活率、菌负荷、抗体滴度、补体/吞噬相关功能和组织病理。

如果动物实验资源不足，先完成体外替代终点：鱼类血清抗体结合、补体沉降、吞噬促进、巨噬细胞刺激后的细胞因子和抗原呈递相关指标；这些结果必须标为“免疫原性/功能证据”，不能直接写成保护性效力。

## 5. 成本核算框架

以下为**预算模型而非机构报价**。实际价格需用本地平台报价替换。

| 项目 | 计算方式 | 示例数量 | 单价变量 | 小计 |
|---|---|---:|---:|---:|
| Tn-seq 文库/测序 | 菌株 × 条件 × 重复 | 2×2×3=12 | `C_Tn` | `12C_Tn` |
| CRISPRi-seq 文库/测序 | 菌株 × 条件 × 重复 | 12 | `C_Cri` | `12C_Cri` |
| RNA-seq | 菌株 × 条件 × 重复 | 12 | `C_RNA` | `12C_RNA` |
| 菌株培养与质控 | 菌株 × 条件 × 批次 | 按平台报价 | `C_culture` | `N*C_culture` |
| sgRNA/载体构建 | 候选 guide 数量 | 10–30 候选 | `C_guide` | `N*C_guide` |
| qPCR/表型验证 | 候选 × 条件 × 重复 | 5–10 候选 | `C_qPCR` | `N*C_qPCR` |
| 蛋白表达纯化 | 候选数 | 5–10 | `C_protein` | `N*C_protein` |
| 免疫与攻毒 | 组别 × 鱼数 | 需伦理审批 | `C_animal` | `N*C_animal` |
| 计算与存储 | 原始数据 TB × 月 | 按云/本地报价 | `C_storage` | `TB*C_storage` |

总预算为上述小计之和，加 10%–15% 预备费。最先削减的应是“重复性预测模型数量”，不应削减生物重复和阴性/阳性对照。

## 6. 里程碑与失败处理

| 月份 | 里程碑 | 失败时的替代 |
|---|---|---|
| 0–2 | 数据下载、审计、菌株和条件预实验 | 缩小物种集合，保留完整审计 |
| 2–5 | Tn-seq/CRISPRi-seq 文库和测序 | 先完成 RNA-seq 和独立 CRISPRi |
| 5–7 | M1/M2 更新与候选排序 | 使用冻结模型和条件化线性基线 |
| 7–10 | 候选因果验证 | 增加不确定性采样，不扩大候选总数 |
| 10–14 | 抗原与体外免疫验证 | 延后动物实验，先完成替代终点 |
| 14–18 | 保护性验证与外部菌株测试 | 报告适用边界，不把体外结果外推为保护性 |

## 7. 结论

VaccineGPT 最有价值的创新不是“用更多模型预测更多分数”，而是把跨数据库证据、条件化组学、同源安全评估和可执行实验设计统一为一个可追溯闭环。近期优先级应为：完成下载失败原因审计与可替代来源、运行 self-check、建立 E. coli/E. piscicida 的条件化多组学基线、用独立实验重复验证 T1/T2，再逐步引入 T3 和动物保护性终点。

## 参考文献

1. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*. 2023. DOI: 10.1126/science.ade2574.
2. Elnaggar A, et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. *IEEE TPAMI*. 2022. DOI: 10.1109/TPAMI.2021.3095381.
3. Ji Y, et al. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language. *Bioinformatics*. 2021. DOI: 10.1093/bioinformatics/btab083.
4. Zhou Z, et al. DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. arXiv:2306.15006, 2023.
5. Lee J, et al. RNA-FM: a deep learning model for RNA sequence and structure. *NAR Genomics and Bioinformatics*. 2022. DOI: 10.1093/nargab/lqac012.
6. Langridge GCM, et al. Simultaneous assay of every Salmonella Typhi gene using one million transposon mutants. *Genome Research*. 2009. DOI: 10.1101/gr.094755.109.
7. Chen L, et al. VFDB 2016 update. *Nucleic Acids Research*. 2016. DOI: 10.1093/nar/gkv1239.
8. Vita R, et al. The Immune Epitope Database (IEDB): 2018 update. *Nucleic Acids Research*. 2019. DOI: 10.1093/nar/gky1006.
9. Szklarczyk D, et al. The STRING database in 2023. *Nucleic Acids Research*. 2023. DOI: 10.1093/nar/gkac1000.
10. Zhang R, et al. DEG: a database of essential genes. *Nucleic Acids Research*. 2004.
11. TRANSIT documentation: https://github.com/mad-lab/transit
12. ESM repository: https://github.com/facebookresearch/esm
13. RNA-FM repository: https://github.com/ml4bio/RNA-FM
