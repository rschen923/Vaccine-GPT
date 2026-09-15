# E. coli 与 Edwardsiella piscicida 组学—疫苗候选实验设计

## 核心假设

在标准培养条件下的必需性、鱼类宿主相关压力下的条件适应性、毒力关联和抗原可及性具有部分重叠但不等价的基因集合。多组学一致性加上独立因果验证可以显著降低单一数据库标签噪声，并为 M1/M2 提供可校准的本地证据。

## 实验分层

### A. 预实验和质量控制

1. 对所有菌株做全基因组测序或至少获得可靠参考基因组，统一 locus tag。
2. 记录培养基、温度、盐度、氧条件、菌龄、接种量和生长相位。
3. 先做生长曲线和血清/鱼类体液耐受范围，选择不造成全群体崩溃的压力水平。
4. 将批次、操作者、文库批次和测序 lane 纳入随机化或区组设计。

### B. Tn-seq

- 每个菌株×条件至少 3 个独立生物重复。
- 以非选择培养作为基线，以鱼类宿主相关压力作为对照条件。
- 建库前检查插入位点分布和文库复杂度；测序深度由基因组大小、预期插入密度和最低基因覆盖反推。
- 采用 gene-level effect、FDR、重复一致性和条件交互项建立 T1 证据。

### C. CRISPRi-seq

- 每基因至少多个独立 sgRNA；加入 non-targeting、essential positive-control 和 non-essential control。
- 分别记录 knockdown 效率，避免把 guide 失效当作基因非必需。
- 对出现强极性影响的操纵子进行单独建模；优先选择独立 guide 方向一致的候选。

### D. RNA-seq

- 每个菌株×条件至少 3 个生物重复，必要时增加感染/应激时间点。
- 以 `batch + condition` 为设计矩阵；对 Tn-seq/CRISPRi-seq 发现的候选进行 targeted validation。
- 将差异表达用于机制解释、通路聚类和反馈采样，而不是直接作为 T1 真值。

### E. 候选验证

每个模型置信度层级选择候选：

1. Top-confidence：预测高、跨 assay 一致；
2. Disagreement：不同模型或视图分歧大；
3. OOD：序列/菌株与训练集不同；
4. Negative control：预测低但文献有强证据或反向证据。

优先进行独立 CRISPRi、回补、竞争生长、血清耐受、定位、表达和重组蛋白质量验证。

## 统计与决策

- 主要效应：gene-level log2 fold-change、条件交互项和 FDR；
- 重复性：样本相关性、批次方差比例、独立 guide 一致性；
- 候选保留：至少两类独立证据支持，或单一强证据且机制明确；
- 模型更新：严格按实验批次和菌株分组加入训练，不回填测试集；
- 失败判定：先检查文库覆盖、guide 质量、培养条件和样本混淆，再否定生物学假设。

## 资源与成本

将本文件中的 `N_samples × C_library × C_sequencing` 替换为本地报价。预算中单独保留文库失败重做、菌株全基因组测序、计算存储和生物安全/动物伦理相关费用。建议预备费 10%–15%。

## 输出给 VaccineGPT 的数据

每个基因输出：

```json
{
  "gene_id": "...",
  "organism": "E. piscicida",
  "strain": "...",
  "condition": "...",
  "assay": "Tn-seq|CRISPRi-seq|RNA-seq",
  "effect_size": 0.0,
  "fdr": 0.0,
  "replicate_consistency": 0.0,
  "evidence_level": "L1|L2|L3|L4",
  "source_batch": "...",
  "lineage": {
    "model_version": "...",
    "feat_version": "...",
    "label_version": "...",
    "batch_id": "..."
  }
}
```
