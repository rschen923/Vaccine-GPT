# VaccineGPT 完整讲解报告 261007

> **读者：**具备生物学训练、正在学习计算生物学/机器学习的研究者。  
> **状态：**阶段性实现讲解；10 轮 OA 全文 retrieval 与 ≥20-codebase 检查配额已达到。
> AI 子代理对 90 个随机轮次引用行（89 个去重 campaign works）的选定段落做了抽样复核；
> 此前另有 6 个来源经节选审阅，其中 4 个 campaign works 不在随机样本、2 个 arXiv
> 来源不在 campaign，合计 93 个 campaign works、95 个去重 works。round 5 被安全过滤
> 阻断，其余正文未逐篇主张核查，故不是
> “2,000 篇全文逐篇深读”的终稿。  
> **阅读顺序：**先读 `VaccineGPT_架构报告_261007.md` 的输入、输出和限制，
> 再读本报告。证据状态见 `research_evidence_261007.md` 与
> `gcdto_campaign_261007_summary.json`；十轮索引见
> `iterations_261007/automated_rounds/`，AI 辅助抽样全文节选审阅见
> `iterations_261007/iteration_XX_manual_review.md`。

## 0. 先给结论：现在能做什么，不能做什么

当前代码能做三件工程上的事：

1. 接受蛋白残基表示，计算公开轨、私有轨和 GCE 的向量；
2. 用给定的 PACE、H1/H2/H4、图边和安全旗标，按明文公式计算区间、
   Ret、Safe、Pareto tier 和 AEP 字段；
3. 审计数据文件，逐条保存来源、重复、冲突、序列类型和证据等级。

当前代码**不能据此说某个基因能保护鱼类免疫，也不能输出已验证的减毒靶点**。
已有一个 1,000 条真实序列、one-hot 输入的自监督 M1 smoke 检查点；它不是
ESM-2/CLEF 预训练模型，也没有 target-specific labels。当前没有用于生物学推理的
VaccineGPT 训练检查点、ESM-2 权重、目标菌 EIB202 私有 CRISPRi/RNA-seq
锚点，也没有可直接用于目标菌分组评估的 MMseqs2 90% 聚类。没有这些输入时，
正确答案是“未知/未排序”，而不是猜一个分数。

## 1. 输入究竟是什么：DNA、基因、蛋白和表位不是同一种东西

### 1.1 DNA 与蛋白序列

细菌染色体是 A/C/G/T 字母构成的 DNA。一个编码区经过转录、翻译后产生
蛋白质。基因预测器会给出 ORF（开放阅读框）、坐标、链方向、可能的操纵子
和蛋白序列。v1 的 M1 输入是蛋白氨基酸序列，不是原始 DNA；这样才能把
蛋白质语言模型 ESM-2 的每残基表示接入 CLEF 风格编码器。

DNA 仍然重要：坐标和邻居关系能构造基因组图；但把 DNA 直接塞入 v1 的
蛋白编码器会改变模型假设，所以代码契约拒绝这种混用。只有 v2 才用 Evo 2
处理 DNA 窗口。

### 1.2 IEDB 表位肽不是 ORF 蛋白

IEDB 记录中的短氨基酸片段可以是免疫测定用的表位肽，而不是细菌全长蛋白。
它们应保留为 `epitope_peptide` 类型，不能与 VFDB 全长蛋白混成同一输入集。
此次 public-final 重审把两种序列分开：30,173 条 VFDB 蛋白和 114,542 条
IEDB 表位记录。M3-S 可在未来读取表位信息，但表位结合、免疫原性和保护效力
是三个不同问题。

### 1.3 ESM-2、ESM-3、Evo 2 的角色差别

- **ESM-2**：蛋白序列模型，v1 的默认冻结残基表示来源。官方 fair-esm
  `esm2_t33_650M_UR50D` 通常给 1280 维残基表示。代码从序列生成 `[B,L,1280]`。
- **ESM-3**：可同时处理蛋白序列、结构、功能等轨迹的生成/多模态模型。
  它是可选比较器，不自动变成 H1-H4 分类器。Biohub 的小型开放权重仍较大，
  本机未下载/运行。
- **Evo 2**：核苷酸/DNA 序列模型，供 v2 全基因组 DNA 窗口使用。官方 README
  指明 Linux/WSL2 与 CUDA 运行约束；不是当前 Windows CPU 版 PyTorch 的本机组件。

三种模型的输出需要分别版本化。训练目标、训练语料和输出维数不同，不能因
名称相近而互换或宣称等价。

## 2. 为什么用对比学习

同一个蛋白可以有多种信息描述：氨基酸序列、功能注释文本、结构、进化保守性、
RNA-seq 或 PACE。对比学习不一定需要“这个基因是不是疫苗靶点”标签，它先学
“同一个实体的两种描述应相近、不同实体应相对分开”。

令序列塔输出 \(z_i\)，同蛋白某公共模态输出 \(z_i^+\)，正样本是同一蛋白的
配对；batch 内其他蛋白构成负样本。余弦相似度为：

$$
\mathrm{sim}(u,v)=\frac{u^\top v}{\|u\|_2\|v\|_2}.
$$

InfoNCE 将正对与负对写成 softmax 对数似然：

$$
\mathcal L=-\frac{1}{N}\sum_{i=1}^N
\log\frac{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)}
{\sum_{j=1}^N\exp(\mathrm{sim}(z_i,z_j^+)/\tau)}.
$$

温度 \(\tau=0.1\) 是项目初值；不是说 CLEF 的论文一定采用这个数值。若
负样本其实是生物学同类、只是标签/ID 不同，它就是假负样本，会误导表示空间。
这正是需要精确去重、同源簇和来源 provenance 的原因。

### 2.1 公共轨与私有轨为什么分开

公开数据库量大但噪声和跨物种迁移多；私有实验量小但可能是目标菌的真实测量。
如果两个来源直接混成一池，公开标签可能覆盖私有实验。代码让它们使用独立
投影/损失，再到 M2 的决策核汇合：

- 公共轨学习公开蛋白/注释/结构的关系；
- 私有轨学习序列与真实组学信号的对应；
- 私有负样本仅在同一基因组内采样；
- 缺失组学用 mask 标示，不能把缺失静默当 0。

当前 14 维私有输入只定义了 schema：PACE 3 维、RNA-seq 8 维、可选 ECP 3 维。
本地数据中没有经核实的 EIB202 私有锚点，因此 smoke test 中 `private_available`
为 0。

## 3. 从蛋白表征到基因组上下文

只看单条蛋白序列不知道它相邻基因、是否同操纵子、是否与其他蛋白互作。
GCE（Genome Context Encoder）将节点表征按图邻居聚合。每个注意力系数是
邻居消息相对于当前节点的权重，GATv2 两层输出新的 128 维 `z_gctx`。

当前边型包括操纵子、共表达、共适应和互作。每条边都带证据级：

- 公共数据库/预测边通常为 E3；
- 目标菌私有实验产生的共表达/共适应边才可能为 E2；
- 预测边不能在报告中伪装成实验边。

邻居边构造与图拆分同样重要。将同一同源簇、同一菌株的相似基因分到训练
和测试两侧，会让模型“见过答案的近亲”。MMseqs2 90% identity 聚类尚未
安装/执行，所以现在不能宣称图评估没有同源泄漏。

## 4. 从证据到标签：E1-E4 不能从数据库名字推断

| 级别 | 当前定义 | 例子与限制 |
|---|---|---|
| E1 | 目标菌/目标条件的直接实验与明确文献来源 | 需能从行记录追到实验和 primary reference |
| E2 | 被明确标记为私有的实测数据 | 如目标菌 CRISPRi、RNA-seq、蛋白组；“文件在本地”本身不够 |
| E3 | 公开计算或预测证据 | 同源预测、模型打分、STRING 等 |
| E4 | 文献外推、争议、物种不匹配或无法追溯的来源记录 | 保留用作弱监督或待人工复核，不作金标签 |

在 public-final 文件中，旧字段把 VFDB 标为 L1、IEDB 标为 L2，但样本行缺少
可核验 assay/primary reference/目标菌信息。新的审计器不将 L1/L2 自动翻译成
E1/E2；这些行暂按 E4 弱监督。`training_role` 与 label provenance 分开记录。
若同一 gene/task 有相反标签，所有该组来源行一起进 `quarantine_conflict`，
不能平均成一个数或优先保留最后一行。

## 5. 数据审计是怎样找出问题的

### 5.1 精确重复与同源性

SHA-256 对规范化序列可识别**精确序列重复**；它不能发现 90% identity 的
同源序列，也不能证明两个序列代表同一 locus。此处的每一步应分别输出：

1. 文件内重复 gene/task 组；
2. 跨文件精确重复序列组；
3. MMseqs2 同源簇；
4. 基因组/locus/操纵子身份；
5. assay/source 标签冲突。

这五种问题不能用一个“删除重复”按钮统一处理。完全重复的序列可以共享
embedding cache，但它的来源标签和不同菌株背景仍要保留。

### 5.2 这次本地重审得到的结果

对 `processed/public_final/sequences.jsonl` 与 `labels.jsonl` 的独立重算发现：

- 两侧各 144,715 行，行级 ID 可联结；无 orphan、无精确重复序列、无冲突组；
- 30,136 个 VFDB ID 可通过唯一序列哈希回到原始 FASTA header；
- 36 个 VFDB ID 不能唯一解析，保留原始 ID 并标记复核，不猜 accession；
- 114,542 个记录识别为表位肽；3,690 个超出本审计使用的常见长度复核范围；
- 228 个含歧义氨基酸；这些是 review 标志，不是静默改写；
- 保守证据审查后，当前此处可作为 E1/E2 gold 的标签数为 0。

全 processed label inventory 有 22,739 个 `*labels.jsonl` 文件、321,732 行。
重复和冲突统计来自每个文件内的 gene/task 分组。`public/labels.jsonl` 中
发现 7,958 个重复 gene/task 组和 26 个相反标签冲突组；这是聚合旧数据的
内部质量问题。输出副本保留所有行，并将 26 组 quarantine。

另一个历史 `sequence_index.sqlite` 的 SQLite integrity check 报页号错误。
审计不对该原始索引做修复或覆盖；本次的 exact-sequence index 用独立的新文件
重建。即使该 exact index 可用，也不代替 MMseqs2 同源评估。

## 6. M2 如何把分数变成可审计决定

### 6.1 PACE 与减毒强度

PACE 将 infection-time fitness 拟合成 \(a t^2 + b t + c\)。负斜率 \(b\)
可表示 fitness 随时间下降的信号，但“负斜率”受菌株、时间点、实验条件和
fitness 定义影响。拟合少于 3 点、设计矩阵秩不足或 metadata 不全时输出
`not_fitted`。

M2 的设计点值是 H1 预测与 `norm(-b)` 的均值。`norm` 必须按同一全基因组
的 beta 值 min-max；若一条记录没有 PACE 或 H1，不得只用现成那个分量给出
假精度。于是 Att 输出一个 conformal 区间；校准集不够时区间变宽，校准集
没有时回到 `[0,1]`。

### 6.2 抗原保留 Ret

删除基因不仅影响自身蛋白，也可能影响操纵子邻居或互作网络。中心性惩罚
与邻居抗原分数共同近似这种风险。公式里的两个邻居均值相乘：操纵子风险
与互作风险都高时，保留量下降。互作数据如果是 STRING/模型预测，只能是 E3，
不能把图边加权后就升格成实验。

### 6.3 Safe、Pareto、三器

四个旗标是 MGE、AMR、相变异和毒性。只有四类都实际检查过，才能算 Safe；
缺少任何一类就返回未评估。H4 核心必需概率大于 0.5 时硬否决。

Pareto 前沿只比较 Att 与 Ret 两个目标，不把主观权重混成唯一分数。Safe
进入层内效用和警示。三器分别是位置注意力、少样本原型和决策贡献分解；
三器不可用时就没有共识，不可用随机初始化权重造出一个“2/3 通过”。

## 7. 一千条真实序列运行到底测了什么

样本取自 144,715 行 public-final 中经 raw VFDB 精确映射、未标成身份歧义的
蛋白记录。随机 seed 为 20261007，样本为 1,000 个真实序列。脚本将其转成
氨基酸 one-hot smoke 表示，逐批送入 M1、GCE、H1-H4 随机初始化前向，并输出
九字段 AEP 格式：

- M1 `z_pub/z_exp/z_gctx`: `[8,128]`；
- position scores: `[8,256]`；
- H1-H4 输出：每批 8 个 logits；
- 共 1,000 行通过 M1/M2 forward。

**没有**载入 ESM-2、CLEF checkpoint、预训练 M1/M2 权重、EIB202 私有模态
或真实基因组图。AEP 所有 1,000 行都 `unranked`，Att 区间 `[0,1]`、
Ret/Safe `null`，毒力和保留抗原列表为空。该运行验证的是数据/接口/批处理
可通过，不是“全模型已训练”，更不是生物学效果验证。

## 8. 如何复现工程测试

从仓库根目录运行：

```powershell
.\VaccineGPT\.venv\Scripts\python.exe -m compileall -q VaccineGPT
.\VaccineGPT\.venv\Scripts\python.exe -m unittest discover -s VaccineGPT\tests -v
.\VaccineGPT\.venv\Scripts\python.exe VaccineGPT\scripts\run_m1_demo.py
.\VaccineGPT\.venv\Scripts\python.exe VaccineGPT\scripts\run_m2_demo.py
.\VaccineGPT\.venv\Scripts\python.exe VaccineGPT\scripts\run_end_to_end.py
```

所有 demo 输出中的 `"mode"` 都标为 synthetic/smoke；不得引用为训练指标、
验证集准确率或疫苗候选排名。

### 8.1 真实 1,000 序列自监督训练

训练用固定 SHA256 identity 拆分 70/15/15，防止精确重复跨 train/validation/test；
目前不能防止同源蛋白跨 split，因为全库 MMseqs2 尚未完成。训练正对为同一
protein 的两个重叠窗口，优化公开序列编码器与 GCE 自环路径的 InfoNCE。该
预任务测试模型能否学会相似视图一致性，不是 H1-H4 标签预测。最佳验证损失
选第 4 epoch；heldout crop retrieval Top-1 在初始化时已为 1.0，所以该测试
没有辨别力，不能作为模型性能提升证据。

后续 M2 训练尝试要求 E1/E2 标签和 `mmseqs2_90` group provenance。用 E4
弱标签会被训练函数拒绝。此拒绝是数据/监督契约按预期工作，不是训练失败。

## 9. 阶段门如何判定

在要宣称 G-A/B/C/D 前必须有：

1. 精确版 ESM-2 与 CLEF 权重 provenance；
2. target-species genome/ORF 和源库版本；
3. MMseqs2 identity cluster 与按 cluster 分割；
4. 每个任务真实、row-level 的实验标签、negative/unknown 语义和 assay 条件；
5. 从不接触训练的 calibration/test；校准符合交换性/域迁移限制；
6. H1-H4 真实训练、基线、消融、跨物种分层和错误分析；
7. GAT edge 证据、网络连通性和去边消融；
8. AEP 每行证据级、风险旗标、unknown 附录和独立人工复核。

这些门没过时，系统可报告“工程流程可执行”，不能报告“实现预设疫苗目标”。

## 10. 术语快速表

| 名词 | 白话解释 |
|---|---|
| residue embedding | 蛋白每个氨基酸位置的一串数字，表示模型从序列上下文得到的向量 |
| InfoNCE | 配对描述拉近、未配对描述在 batch 中区分开的训练损失 |
| false negative | 实际同类却被当负样本拉远的记录 |
| GATv2/GCE | 根据邻居和边类型更新基因向量的图神经网络 |
| PACE | 将多时间点 fitness 变化压成二次曲线系数 |
| conformal interval | 用单独校准集残差构造区间；样本/分布条件不满足时不能保证覆盖 |
| Pareto front | 多目标下不被其他候选同时支配的集合，不等于唯一“赢家” |
| E1-E4 | 来源证据等级，不等于模型置信度 |
| exact duplicate | 序列完全相同；不能说明同源簇、同一基因或标签有效 |

### 阅读和证据来源

可核验的选定全文节选审阅和 22 个代码库索引见
[`research_evidence_261007.md`](research_evidence_261007.md)。目前的全文证据只
支持方法/限制层陈述；任何针对 VaccineGPT 保护效果或湿实验的结论仍为未知。
