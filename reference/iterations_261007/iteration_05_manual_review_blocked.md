# GCDTO 迭代 05 — 功能基因组数据 / manual-review 状态

**主题：** CRISPRi/TnSeq、fitness trajectories、condition-specific essentiality 与 H1。  
**全文获取：** round 05 的 OA fullTextXML 已下载 200/200；全文索引、访问 URL、
哈希与每轮 reference set 在 `..\data\research_campaign_261007\`。  
**AI 辅助节选审阅：** 本轮抽样复核任务被安全审查阻断，未完成。没有重试同一请求，
也没有使用其他工具绕过该限制。

## 已审查证据边界

在前一轮的 evidence source map 中，Jana et al., “CRISPRi–TnSeq maps
genome-wide interactions between essential and non-essential genes in
bacteria”, DOI `10.1038/s41564-024-01759-x`, PMCID `PMC11371651` 的全文
Introduction、Results “Essential–non-essential interaction mapping with
CRISPRi–TnSeq” 与 Fig. 1 曾被打开并概述：研究在 *Streptococcus pneumoniae*
采用 CRISPRi knockdown essential genes 与 TnSeq knockout non-essential genes
结合。该描述仅作功能基因组学方法背景；没有将其数值结果迁移至 VaccineGPT
或 EIB202，也不包含可据此制备/执行任何生物实验的操作说明。

## 正题 / 反题 / 合题

- **正题（范围有限）：** 时间序列 fitness 与条件性遗传证据可以形成一类与基因
  缺失后适应度相关的监督/校准信号；本项目 PACE 接口仅在真实 timepoint 和
  目标菌测量存在时才拟合。
- **反题：** essentiality 与 antigen accessibility/protective immunity 是不同
  endpoint；不同物种、菌株、assay 或条件的 label 不能直接互换。本地目前没有
  EIB202 的 target-specific E1/E2 PACE/RNA-seq gold。
- **合题：** H1/PACE 作为有 provenance 的目标菌证据输入；缺失/稀疏/设计矩阵
  秩不足则 `not_fitted` 或 unranked。Public computational evidence 不能覆盖
  target-specific measured directions。

## 可证伪检测器

需要由合规、获批的目标菌 functional-genomics 研究产生独立 E1/E2 labels 后，
预注册条件、盲测 split 与模型判据；在没有这些记录时，G-B/H1 仍为 blocked，
模型不输出减毒推荐。
