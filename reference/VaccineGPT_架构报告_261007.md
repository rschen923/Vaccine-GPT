# VaccineGPT 架构报告 261007

> **文档状态：阶段性实现版；不是已通过文献配额与生物学验收的终稿。**
> 生成日期：2026-10-07。实现范围依据随附 v16R 架构报告、目标核验清单
> 与代码树型优化法。公开来源、审阅范围及缺口见
> [`research_evidence_261007.md`](research_evidence_261007.md)、
> [`code_evidence_ledger_261007.jsonl`](code_evidence_ledger_261007.jsonl) 与
> `..\data\research_campaign_261007\normalized_evidence_ledger.jsonl`。
>
> **配额状态：** 已完成 10 个主题轮次，每轮抓取并校验 200 篇可公开获取全文
> 的 Europe PMC XML；已检查 10 轮共 202 个公共代码仓库引用（192 个全局唯一）。
> 但并未逐篇人工阅读全部 2,000 篇论文：AI 子代理对 90 个随机轮次引用行
>（89 个去重 campaign works）的摘要/方法/结果节选做了辅助审阅；此前另有 6 个
> 来源做过节选审阅，其中 4 个 campaign works 不在该随机样本、2 个 arXiv 作品
> 不在 campaign，合计 93 个 campaign works、95 个去重 works。round 5 的代理审阅
> 受安全过滤阻断。因此本报告
> 是“检索配额达标、模型辅助全文主张抽样、目标生物学验收未通过”的阶段版，
> 不把全文抓取量误称为全文逐篇人工深读，也不声称保护性免疫已实现或验证。

## 1. 不动点、范围与证据边界

### 1.1 预设目标

> 细菌全基因组输入 → 模型推理/计算 → 输出能够指导疫苗开发的结果：
> 毒力基因、减毒活疫苗靶点、亚单位疫苗靶点等。

目标按两版兑现：**v1 输入注释蛋白序列，不把 DNA 序列作为模型输入**；
基因位置和基因组图仅作为注释上下文。**v2 才接受全基因组 DNA 窗口**。
两版共享 AEP 输出契约。当前代码覆盖 v1 的结构、M2 数学和数据处理，
并提供 v2 接口/窗口/头部骨架；**实际生物学模型、目标菌验证、v2 Evo2
推理尚未完成**。

### 1.2 交付边界

| 输出 | 代码状态 | 当前能否作生物学结论 |
|---|---|---|
| 靶点行：基因、Att 区间、Ret、Safe、证据级、三器、机制、警示、实施路径 | AEP schema 与决策模块已实现 | 否；无经过训练和校准的 VaccineGPT 检查点 |
| 毒力基因清单 | H1/H3 接口 | 否；当前公开 VFDB 成员资格只作为弱监督，不等于目标菌实验验证 |
| 保留抗原清单 | H2/Ret 接口 | 否；需表达、表面定位和保护证据 |
| 未解读附录 | 数据/输出 schema 已建立 | 可以列出未解析/缺证据项，不能自动补出答案 |
| ξ_ext 声明 | AEP helper 已实现 | 剂量、给药程序、递送、目标人群、公平性和临床试验均不计算 |
| M3-S 亚单位适配 | 只读接口与多价覆盖公式 | 未实现表位预测/保护验证 |
| M3-N 核酸适配 | 只读输入/输出接口 | 未执行密码子优化 |
| v2 DNA/Evo2/H5 | 窗口、节点类型、适配器及 H5 scaffold | 未在本机安装、训练或验证 |

### 1.3 四公理落实方式

- **A0 可追溯：** 每个标签保留来源/行级 provenance；模型/特征/标签/批次
  版本保留 lineage。未知量采用 `null`、`unavailable` 或 E4，不以随机分数
  冒充观测。
- **A1 主线唯一：** v1 主线只有“蛋白 → M1 → GCE → H1-H4 → M2/AEP”。
  M3-S/N 仅只读；v2 才增加 DNA 窗口与 H5。
- **A2 数据层级：** E1/E2 标签方可作为金监督；E3/E4 保留为弱监督；
  异方向证据不平均，冲突组 quarantine。
- **A3 完备性：** AEP 必须含未解读/警告/外部参数状态。无依据时不做候选
  排名。

## 2. 总体数据流

```text
v1：注释蛋白序列（不是 DNA）
  ├─ 冻结 ESM-2 → 每残基表示 [B,L,1280]，L≤256
  ├─ CLEF 风格序列 Encoder A → z_pub [B,128] + 序列位置注意力
  ├─ 公共模态（文本/结构/MSA/PSSM）→ 各自投影 → 公共对比损失
  ├─ 私有组学（PACE 3 + RNA-seq 8 + 可选 ECP 3）→ 独立私有适配器
  │    └─ 负样本只在同一基因组内，观测掩码透传
  └─ z_pub + 私有上下文/覆盖掩码 → 两层 GATv2 GCE → z_gctx
       └─ M2 单一决策核：H1/H2/H3/H4 + 证据/区间/Pareto → AEP

v2：全基因组 DNA
  └─ Evo2 窗口嵌入 → 元件注释/证据 → GCE 扩展节点与边 → H1-H4 冻结复用 + H5
       └─ AEP v2 仍满足同一九字段契约，实施路径由实测构建数据训练

M3-S / M3-N：只读 M1/M2 与 AEP；禁止向主线回写。
```

## 3. v1 M1 编码段

### 3.1 输入契约

每个样本对应唯一蛋白/基因实体，至少需要：

| 字段 | 类型/规则 |
|---|---|
| `gene_id` | 稳定、可回溯的实体 ID；截断 ID 不得静默拼接 |
| `protein_sequence` | 氨基酸序列；异常字母、歧义残基、短片段单独记录 |
| `genome_id` | 基因组/物种/菌株；私有负样本采样和泄漏检查使用 |
| `split_group_id` | MMseqs2 ≥90% identity cluster 或预注册的 genome holdout |
| `sequence_hash` | 规范化精确序列 SHA-256；不替代同源聚类 |
| `lineage` | 模型、特征、标签和批次版本 |

M1 的主要输入张量是 ESM-2 residue representation `[batch, length, 1280]`
和有效残基掩码 `[batch, length]`。长度截到 256 aa；v1 拒绝把 DNA 序列
作为同一轨输入。PSSM/MSA、3Di、注释文本属于可选公共模态，不存在时不
伪造零向量为真实观测。

### 3.2 ESM-2 与 CLEF 风格 Encoder A

主干采用可选上游 `facebookresearch/esm` 的 `esm2_t33_650M_UR50D`，
默认冻结。其后使用代码中的 CLEF 风格 Transformer：

- 输入线性投影 1280→128；
- 2 层、8 头、FFN 128、dropout 0.1、pre-LN；
- 以有效残基数作均值池化；
- 输出 128 维序列表征 `z_pub` 与位置注意力；
- 每个公共模态有独立投影头，不将所有原始特征简单拼接后伪称一条轨。

CLEF 全文中实际审阅的 Fig. 1A-C 和 Results 支持“序列/生物特征对比
编码用于效应蛋白预测”这一方法先例；它**不**证明疫苗保护性抗原预测
或活疫苗安全性。详见证据状态表。

### 3.3 InfoNCE 与公私分轨

对锚点 \(z_i\) 与同一实体的正样本 \(z_i^+\)，使用余弦相似度：

$$
\mathcal{L}=-\frac{1}{N}\sum_i
\log\frac{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)}
{\sum_j\exp(\mathrm{sim}(z_i,z_j^+)/\tau)},\qquad
\mathrm{sim}(u,v)=\frac{u^\top v}{\|u\|_2\|v\|_2},\quad \tau=0.1.
$$

公共模态使用公开批次内的负样本；私有轨只允许同一基因组内的非配对
样本成为负样本。私有 feature mask 缺失时 `private_available=false`，
不将“没测”当成“测量值为 0”。该代码实现只更新轻量投影/适配器，
不更新 ESM-2 主干。

### 3.4 私有模态

目标定义总维数为 14：

- PACE 二次拟合 `[a,b,c]` 共 3 维；
- RNA-seq 4 组比较的 `[log2FC, -log10(p)]` 共 8 维；
- 可选 ECP 3 维，并带逐项缺失掩码。

数值只在存在对应 assay、样本元数据与 target organism provenance 时才可
升级为 E2。当前 processed public-final 不含上述 EIB202 私有实验锚点。

## 4. 基因组上下文 GCE

GCE 使用两层 GATv2，维数 128，4 头，自环、残差、LayerNorm，LeakyReLU
负斜率 0.2。每条边保留类型、权重、来源和证据级：

| v1 边 | 判据 | 默认来源级 |
|---|---|---|
| 操纵子 | 注释为同操纵子；或同链相邻 ORF、间距≤60 nt | 公开注释/预测，须与实验边区分 |
| 共表达 | 私有 RNA-seq 向量 `abs(r)≥0.8`，权重 `r²` | E2，仅已测样本 |
| 共适应 | PACE `[a,b,c]` 余弦相似≥0.9 | E2，仅真实 PACE |
| 互作 | CLEF-EEI≥0.5 或 STRING≥0.7 | E3，上限为预测证据 |

v2 预留 `dna_regulation` 和 `dna_element_proximity` 两槽。没有实测/注释
要素时，窗口标为 `unclassified_window`，不静默命名成启动子或操纵子。
`in_silico_knockout` 对比删节点前后 GCE 表征，输出邻居位移均值；这是
模型敏感性量，不是实验因果效应。

## 5. v1 M2 决策核与 AEP

### 5.1 任务头

H1=体内定植/必需性；H2=表面/抗原证据；H3=效应/分泌；H4=体外核心必需性。
每个 head 是共享 128 维输入上的线性 logits。不存在对应 assay/金标签时
该 head 的监督状态为 unavailable；不能把未标注基因自动填负例。

### 5.2 PACE、Att 与 conformal 区间

对归一化感染时间 `t/t_max` 与 fitness `phi` 作二次最小二乘：

$$
(a,b,c)=\arg\min_{a,b,c}\sum_m[\phi_m-(a t_m^2+b t_m+c)]^2.
$$

少于 3 个时间点或设计矩阵秩不足时返回 `not_fitted`。减毒点值设计式：

$$
\widehat{Att}(g)=0.5\,\widehat{S}_{H1}(g)+0.5\,\mathrm{minmax}(-b_g).
$$

`minmax(-b)` 必须在同一评估基因组总体上计算，不能逐单基因计算。校准
残差用 split conformal 的有限样本分位数；`n_cal<100` 时以最大残差外加
有限样本膨胀，`n_cal=0` 或模型分量缺失时返回 `[0,1]` unavailable 区间。
目标置信度 90% 是项目默认值，不是本机验证通过的保证。

### 5.3 Ret、Safe、Pareto 与组合

H2 概率 `A(g)`、操纵子邻居 `O(g)`、互作邻居 `N(g)`、归一化介数中心性
`c_bar(g)` 给出：

$$
Ret(g)=e^{-2\bar c(g)}
\cdot\frac{A(g)+\sum_{u\in O(g)}A(u)}{1+|O(g)|}
\cdot\frac{A(g)+\sum_{v\in N(g)}A(v)}{1+|N(g)|}.
$$

注意：两个邻居平均在公式中相乘。没有 H2、图或中心性输入时 Ret 为
`null`，不是零，也不是从蛋白序列臆测的表面表达。

四类安全旗标完整可判时：

$$Safe(g)=1-\min(1,0.2\,n_\mathrm{flag}(g)).$$

少一类旗标就返回未评估；H4 概率超过 0.5 才触发硬否决。Att/Ret 上做
Pareto 前沿：前沿且两者≥0.6 并且三器≥2/3才是 Tier A；前沿其他为 B；
被支配为 C。标量 `U=Att×Ret×Safe` 仅用于层内排序。旗标、冲突或缺失
不删除候选，必须写入 `degrade_reasons`。

组合靶点最多 3 基因，候选先按 U 截前 60；Att 组合按独立性假设
`1-product(1-Att)`，区间按成员端点传播并明确标成“非独立校准”。Ret
按删除集合及邻域剩余抗原分数重算。

### 5.4 输出契约

九字段每行：

`gene/product | Att interval | Ret | Safe | evidence E1–E4 | three-reader |
mechanism | warning | implementation path`

v1 `implementation_path` 恒为空。三器是：序列注意力位置排序、私有空间
原型相似度排序、`Att/Ret/Safe` 对数贡献。每个器缺数据则标记未运行，
不可由随机 logits 补齐。

ACQ 用共形半宽、三器分歧、到 A 阈值的距离分别全基因组 min-max 后相乘。
没有有效 calibration / reader 时不得输出“实验优先级预测”。

## 6. v2 与 M3

`src_m1/genome_v2.py` 支持受控 DNA 窗口 schema；Evo2 embedding projector、
GCE 的新增节点槽和 H5 头是实现 scaffold。Evo2 公开 README 要求 Linux
或 WSL2/CUDA，当前 CPU PyTorch/4GB T600 配置不适合本机推理。ESM-3 使用
Biohub 的可选 adapter；公开 ESM3 是生成/多模态模型，不是 v1 分类器，
且本机未安装权重/依赖。

M3-S 输出呈递、免疫原性、保护三层字段分开；多价覆盖仅在 presence、
expression、P(ON) 区间与血清型权重齐备时计算。M3-N 只校验 CDS/宿主密码子
表接口。本两模块只读，不能修改 M1/M2。

## 7. 数据与标签检查结果

源数据位于仓库外的 `..\data\`。所有源数据保持不变；清洗产物是新副本。
`audit_and_clean_data.py` 对 `processed/public_final` 的 144,715 个
sequence/label 记录独立重算：

- 30,173 条 VFDB T2 记录（蛋白），114,542 条 IEDB T3b 记录（表位肽），
  两类序列不混为同一实体类型；
- 唯一序列/ID 联结零 orphan，精确重复序列组 0，gene/task 冲突组 0；
- 根据 raw VFDB FASTA 的**唯一精确序列**恢复 30,136 个原始 accession；
  36 个 accession 未能唯一恢复，仍 quarantine，不猜补；
- 130,097 个 positive 与 14,618 个 negative 值；所有未决是否属于目标菌/目标
  assay 的标签按 E4 弱监督处理，当前 gold-eligible E1/E2 行为 0；
- 114,542 条 IEDB 记录为 peptide role；短/歧义/超常长度项留审计状态。

全 processed label inventory 覆盖 22,739 个 `*labels.jsonl` 文件、
321,732 行、107,295,284 字节。旧 `processed/public/labels.jsonl` 中有
7,958 个文件内重复 gene/task 组和 26 个不同方向标签冲突组；这些冲突已在
新 evidence-tiered copy 中逐行 quarantine，未平均覆盖，共 72 条冲突证据行。
321,732 行重新分级均为 E4 弱监督；该统计是文件内计数求和，不等于跨文件
全球唯一实体数。

**不能称已 training-ready：** 行级原始 assay/reference/target-strain 证据
不足；公开 final 没有可靠 genome/operon/MMseqs2 cluster split；EIB202 私有
实验金标签缺失；现存 `sequence_index.sqlite` 的 SQLite integrity check
失败，同目录 `.tmp` 的 `seen` 表为 0 行。全 processed sequence JSONL 共约
100,286,044,997 bytes；本轮完整 exact-sequence 扫描未能在当前执行预算内
完成，366,919,680-byte 部分索引已丢弃，全库 exact duplicate 数为 unknown。
状态记录在 `..\data\processed\vaccinegpt_complete_audit_261007\sequence_audit_status.json`。
MMseqs2 90% 同源聚类也尚未运行；精确重复与同源泄漏是两件事。

## 8. 代码映射

| 层 | 主要文件 |
|---|---|
| ESM-2/ESM-3/Evo2 adapters | `src_m1/encoders/foundation.py`, `src_m1/encoders/factory.py` |
| CLEF-style encoder / dual track | `src_m1/models.py`, `src_m1/training.py` |
| GATv2 / graph / genome windows | `src_m1/gce.py`, `src_m1/graph.py`, `src_m1/genome_v2.py` |
| H1-H4/H5 | `src_m2/models.py`, `src_m2/h5.py` |
| PACE / conformal / graph metrics / Pareto / ACQ | `src_m2/decision.py` |
| Calibrated-model gate and decision→AEP wiring | `src_m2/pipeline.py`, `src_m2/aep.py` |
| M3 one-way contracts | `src_m3/interfaces.py` |
| Evidence, exact duplicates, cleaning | `shared/data_quality.py`, `scripts/audit_and_clean_data.py` |
| ESM-2 embedding cache / M1 and M2 training | `scripts/extract_esm2_embeddings.py`, `scripts/train_m1_contrastive.py`, `scripts/train_real_pipeline.py` |
| Public FASTA/IEDB parser | `scripts/prepare_public_sources.py` |
| Raw VFDB/IEDB preparation | `scripts/prepare_public_sources.py` |

## 9. 预设目标核验

| 核验项 | 状态 | 理由 |
|---|---|---|
| v1 不接受 DNA 输入 | **通过（代码契约）** | M1 接收 residue embedding；UDC feature contract 拒绝 DNA on v1 |
| M1/GCE/M2 tensor contract | **通过 smoke** | 20 个单测及合成前向通过 |
| 1,000 个真实蛋白记录通过训练与推理路径 | **工程训练/集成通过，生物学未测** | one-hot 自监督 M1/GCE best checkpoint 已训练和回载；M2 四头通过 forward；1,000 行仍 unranked |
| AEP 四件套与未知态 | **schema 有；真实排名未通过** | M2 无 E1/E2 gold 训练；门控输出 1,000 行 unranked |
| G-A/B/C/D、A1-A9 | **未验证** | 无正式 CLEF reproduced metrics、EIB202 实验锚、盲测/校准集 |
| 10×(200 full-text + ≥20 repositories) 检索引用配额 | **通过（全文抓取/代码仓库检查）** | 每轮 200 OA FullTextXML；202 仓库引用、192 唯一；90 个轮次引用行（89 个 unique works）由 AI 子代理审阅节选；连同此前审阅的来源共 93 个 campaign works / 95 个 unique works，round 5 被安全门阻断，全部2,000篇未逐篇手读 |
| 全部数据已变成目标模型可训练金标签 | **未通过** | E1/E2 可用标签 0；存在 legacy source provenance 和分组缺失 |

### 本机执行边界

本机 `torch.cuda.is_available()==False`；虽检测到 T600 4GB 显卡，当前
PyTorch 是 CPU 版。`fair-esm`、Biohub `esm`、`transformers`、`evo2`
依赖及 ESM2-650M/CLEF checkpoint 均未安装/未提供。真实 1k 跑通验证的是
清洗、张量接口、M1/M2 前向和 AEP unknown contract，不是 ESM2 模型效果或
保护性疫苗目标。

全量 M1 自监督训练结果和测试指标见
[`VaccineGPT_训练报告_261007.md`](VaccineGPT_训练报告_261007.md)。结果显示
验证 InfoNCE 最低点在 epoch 4，但 self-supervised test retrieval 已在训练前
达 1.0，不能作为生物学泛化指标；因此暂未基于它修改 H1-H4 决策结构。

## 10. 十轮研究的通过范围与剩余门

每轮已生成独立 query snapshot/raw response、200 个已下载并哈希校验的
fullTextXML 引文，以及不少于20个有README/实现路径/license状态的代码库条目；
每轮 GCDTO dossier 见 `reference/iterations_261007/automated_rounds/`，AI辅助
抽样全文复核见同目录 `iteration_XX_manual_review.md`。

**未通过/未完成：** 2,000 篇全文没有逐篇人工提取所有方法/数值/限制；
round 5 的 10 篇抽样复核请求被安全过滤阻断，其余9轮的90篇节选经 AI 子代理
复核；100篇抽样状态见 `manual_verification_sample.jsonl`。所有未核查文献的 reliability/
claim-level extraction 都标明 pending/limited，不能把自动关键词支持/反题
标签当成同行审阅结论。即使检索配额达到，本项目仍未通过 G-A/B/C/D、
A1-A9、MMseqs2 同源拆分、目标菌 E1/E2 训练/校准或挑战保护盲测，因此
不满足“预设疫苗开发目标已可实现”的终审条件。
