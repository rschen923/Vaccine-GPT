# VaccineGPT 公开数据获取与目标菌训练结果

**记录日期：**2026-10-07  
**目标菌株：***Edwardsiella piscicida* EIB202  
**数据根目录：**`..\data\raw\public_training_sources\`（位于 Git 仓库之外）

## 结论摘要

已从 ENA 下载并校验两个 EIB202 Tn-seq 项目的全部 6 个 FASTQ 文件，共
5,197,957,373 bytes。文件大小与 ENA 元数据一致，MD5 和本地 SHA-256 均已复核；
每个文件的前 100 条记录通过 FASTQ 结构抽查。下载清单包含 12 个已取得文件
（另含 DEG、Pseudomonas 参考工作簿及 EIB202 基因组），合计 5,220,315,377 bytes。

EIB202 基因组蛋白组的 M1 自监督训练已完成，3,536 条蛋白都能通过已训练 M1
及 M2 接口前向运行。但这**不是**监督生物学训练：目前没有经核验的 EIB202
E1/E2 金标签，M2 未训练、AEP 结果保持 `unranked`，生物学预测状态为
`not_evaluated`。原始 Tn-seq reads 尚未转成经过验证的基因级插入/fitness 标签。

## 已获取的 EIB202 Tn-seq 数据

数据来自 ENA 的两个项目。2019 项目的 ENA 返回 4 个 library rows，其中两个为
RNA-Seq，已排除；只下载两个 `library_strategy=Tn-Seq` runs。2025 项目列出两个
Tn-seq runs，各有一对双端 FASTQ。下表中的 MD5 与 ENA 元数据匹配；文件级 SHA-256
和 URL 等完整来源信息见 `dataset_manifest.json`。

| 项目 / run | 实验条件 | 文件 | 大小（bytes） | ENA MD5 |
|---|---|---|---:|---|
| PRJNA480066 / SRR8208794 | 16°C seawater | `SRR8208794.fastq.gz`（单端） | 657,790,359 | `4f310dcc39f2799884f8b8fe20e2cf05` |
| PRJNA480066 / SRR8208795 | 28°C seawater | `SRR8208795.fastq.gz`（单端） | 536,364,425 | `0f785cae3d9f6983372c0465a9086559` |
| PRJNA1082880 / SRR28236933 | DMEM | `SRR28236933_1.fastq.gz` | 701,220,863 | `5e8f9d6b8b8f983b4a83a0c516f517fe` |
| PRJNA1082880 / SRR28236933 | DMEM | `SRR28236933_2.fastq.gz` | 910,829,889 | `767aefbdb612f7108e07a972dc775217` |
| PRJNA1082880 / SRR28236989 | low-glucose DMEM | `SRR28236989_1.fastq.gz` | 1,046,939,260 | `95bcac5dedc259602e902c8a448c98f6` |
| PRJNA1082880 / SRR28236989 | low-glucose DMEM | `SRR28236989_2.fastq.gz` | 1,344,812,577 | `168e716d192e669b7389a8d7d94ec49b` |

**实验设计限制：**ENA 当前列出的 2019 数据是每个温度一个 run；2025 数据是每个
培养条件一个 run。现有 run 元数据没有证明每个条件存在独立生物学重复。因此，这些
reads 可以作为有来源与条件标记的原始证据保留，但不能仅凭 run 数量建立可靠的条件
差异检验、方差估计或金标签。

抽查检查了每个压缩 FASTQ 的前 100 条记录：记录头以 `@` 开始、分隔行以 `+`
开始，序列与质量字符串长度相同。此抽查不是完整 read-level 生物学 QC，也不识别
转座子连接序列、barcode 或插入位点。

## 必需基因和跨物种参考数据

已取得 DEG 官方下载端点的以下文件：

- `deg_bacteria.csv.zip`：66 个 organism/study 目录条目；这是目录，不是 EIB202
  基因级标签表。
- `deg_annotation_p.csv.zip`：26,619 条多物种注释记录；未发现
  *Edwardsiella* 或 EIB202 匹配记录。
- `DEG10.aa.gz` 与 `DEG10.nt.gz`：各 26,619 条 FASTA 记录；标题中没有
  *Edwardsiella* 或 EIB202。
- Pseudomonas GOLD_84/GOLD_115 参考工作簿：仅为跨物种方法/标签参考，不是 EIB202
  金标签。

因此，DEG 与 Pseudomonas 资源不得被直接转写为 EIB202 essential/non-essential
标签；同源匹配也必须保留物种、基因组、实验条件及证据等级，不能把推断当成目标菌
实测结果。

PMC 中 Wei et al. 2019 的两份补充文件端点返回 proof-of-work 挑战 HTML，而非 PDF
或 XLSX。挑战响应已隔离保留在 `source_responses\rejected\`，清单记录为
`not_acquired` / `rejected_not_a_data_file`；未尝试绕过挑战。

## EIB202 蛋白组与 M1 自监督训练

从 NCBI GenBank `CP001135.1` 的 CDS translation 提取目标蛋白组：

- 3,536 条具有唯一 locus tag 的有效蛋白记录；缺 locus tag、缺 translation、无效
  translation 和重复 locus tag 均为 0。
- 其中有 21 条精确重复序列；训练按精确规范化序列分组，得到 3,515 个唯一序列。
- 蛋白长度范围 37–2,695 aa，中位数 260 aa。
- GenBank SHA-256：
  `8eb9ac33083f1a1be9da71207cd2f9e23a9372d023200a19648465cb07cb1f6a`。
- 蛋白组 JSONL SHA-256：
  `2f9a6b47c720d5e632886fdef875658499a961a45f7958b622e4db713af40fee`。

训练为 M1 序列 crop 一致性自监督任务，不是 essentiality、毒力或疫苗保护分类：

| 项目 | 结果 |
|---|---|
| 输入 | 3,536 条 EIB202 蛋白；氨基酸 one-hot |
| train / validation / test | 2,460 / 527 / 528 个精确序列组 |
| 特征与目标 | 未使用 ESM-2/CLEF 预训练权重、基因图边或 EIB202 H1–H4 标签；训练 two-view crop InfoNCE |
| 最佳 epoch | 5 |
| Validation InfoNCE | epoch 1 为 0.1035；epoch 5 为 0.0440 |
| paired-crop retrieval Top-1 | 训练前与训练后均为 0.9981；该指标接近天花板，只反映同一序列 crop 配对，不表示生物学预测能力 |
| 检查点 SHA-256 | `f636c328d22ccd737cc571ced593f41e8579bfb4a149f61f5bbb33551cdbfadf` |

此 split 只隔离精确重复序列，没有用 MMseqs2 控制 90% 同源泄漏。因此验证集/测试集
结果只适用于这次工程自监督任务，不能作为跨同源簇泛化结论。

已用该检查点让完整 3,536 条蛋白通过 M1→M2 接口，生成 `[3536, 128]` 的
`z_gctx`。M2 checkpoint 未加载，未训练随机头仅用于接口 smoke；3,536 条 AEP
输出均为 `unranked`，`biological_prediction_status=not_evaluated`。详细数值与
检查点路径见仓库外 `..\data\processed\vaccinegpt_eib202_proteome_training_261007\training\`。

此前 1,000 条 VFDB 真实蛋白的自监督工程 smoke 另见
`VaccineGPT_训练报告_261007.md`。VFDB 样本不属于 EIB202，也没有被冒充为目标菌
监督标签。

## 为什么尚未做 Tn-seq 监督训练

FASTQ 下载完成不等于基因级必需性标签已经成立。进行条件 fitness/必需基因分析前，
还需要验证转座子连接端和测序引物结构、确定 barcode/插入序列解析方式、将唯一插入
位点映射到 `CP001135.1`、检查文库复杂度与基因覆盖度，并确认实验重复及其比较设计。
随后才可在保留条件和不确定观察的前提下计算插入计数/fitness，并评估阈值与稳健性。
当前从公开 run 元数据无法确认生物学重复，因此没有声称已经生成这些标签。

此外，Tn-seq essentiality 或特定培养条件 fitness 不等价于抗原可及性、免疫原性或
宿主保护。即使将来得到可信的 Tn-seq gene-fitness 标签，也不能把它直接当成 VaccineGPT
的 E1/E2 疫苗保护金标签。故本轮不做 M2/H1–H4 监督拟合、不输出候选抗原排名；这是
证据门控结果，不是训练代码成功的替代声明。

## 可复核文件

- `..\data\raw\public_training_sources\dataset_manifest.json`：12 个取得文件的 URL、
  accession、物种/条件、大小、MD5、SHA-256、校验状态及限制。
- `..\data\raw\public_training_sources\dataset_manifest.jsonl`：追加式获取记录。
- `..\data\raw\public_training_sources\download_failures.jsonl`：挑战响应拒绝记录。
- `..\data\processed\vaccinegpt_eib202_proteome_training_261007\eib202_proteome.jsonl`：
  带 locus tag 与来源信息的 EIB202 蛋白记录，均标为 unlabelled。
- `..\data\processed\vaccinegpt_eib202_proteome_training_261007\eib202_proteome.report.json`：
  蛋白组提取审计。
- `..\data\processed\vaccinegpt_eib202_proteome_training_261007\training\training_report.json`：
  M1 训练与验证指标。
- `..\data\processed\vaccinegpt_eib202_proteome_training_261007\training\inference_eib202\trained_m1_sample_chain_report.json`：
  全蛋白组 M1→M2 接口与 unranked 输出记录。
