# GCDTO 迭代 05 — CRISPRi / TnSeq / fitness code 核验

**已检查 20/20 公共仓库/代码库**，含 README、相关实现、许可证状态。集合中含 workflow wrappers 和物种专用项目；20 项不是 20 种统计算法。

| # | Repository | 核心路径 | License / 边界 |
|---:|---|---|---|
| 1 | [ioerger/transit](https://github.com/ioerger/transit) | `src/pytransit/analysis/ttnfitness.py` | GPL-3.0；通用 TnSeq 分析 |
| 2 | [ioerger/transit2](https://github.com/ioerger/transit2) | `src/pytransit/methods/ttnfitness.py` | GPL-3.0；另一版本/仓库，不是另一独立方法验证 |
| 3 | [sanger-pathogens/Bio-Tradis](https://github.com/sanger-pathogens/Bio-Tradis) | `bin/tradis_essentiality.R`, `bin/tradis_comparison.R` | README GPLv3 与 `software_license` BSD-like terms 有冲突，先向维护方核实 |
| 4 | [veeninglab/CRISPRi-seq](https://github.com/veeninglab/CRISPRi-seq) | `TINDRi.py` | GPL-3.0；README/source 检查更像 guide-library design/evaluation，未核为通用 counts-to-fitness pipeline |
| 5 | [hancockinformatics/EssentialTnSeq](https://github.com/hancockinformatics/EssentialTnSeq) | `R/et_essential.R` | MIT；处理已产生的 per-replicate calls，不是原始测序到统计的通用全流程 |
| 6 | [mjmlab/pyinseq](https://github.com/mjmlab/pyinseq) | `pyinseq/pipeline.py` | BSD-3；README 有 *V. fischeri* 示例，物种专用边界明确 |
| 7 | [srimam/TSAS](https://github.com/srimam/TSAS) | `TSAS.py` | MIT；mapped read / one-/two-sample analysis |
| 8 | [ruy204/FiTnEss](https://github.com/ruy204/FiTnEss) | `R/FiTnEss_function.R`, `R/callessfun.R` | GPL-3.0；README 示例 *P. aeruginosa* PA14 |
| 9 | [WhiteleyLab/Tn-seq](https://github.com/WhiteleyLab/Tn-seq) | `TnSeqDESeq2_DifferentialsAnalysis.R` | MIT；study scripts，DESeq2 differential fitness |
| 10 | [travis-m-blimkie/TbTnseq](https://github.com/travis-m-blimkie/TbTnseq) | `R/tbt_essential.R` | root LICENSE 未发现 |
| 11 | [KBoehme/TnSeq-Pipeline](https://github.com/KBoehme/TnSeq-Pipeline) | `TnSeq-Pipeline.py` | MIT；mapping/tally tool，不是 essentiality method |
| 12 | [m-jahn/TnSeq-pipe](https://github.com/m-jahn/TnSeq-pipe) | `source/run_tnseq_mapping.sh` | GPL-3.0；barcoded library workflow |
| 13 | [mad-lab/tnseq_GI-python3](https://github.com/mad-lab/tnseq_GI-python3) | `tnseq_GI.py` | GPL-3.0；examples 是 H37Rv/特定 KO 与时点 |
| 14 | [beckham-lab/RB-TnSeq](https://github.com/beckham-lab/RB-TnSeq) | `Scripts/5_BarSeqProc_analyzeExp.py` | GPL-3.0-or-later；BarSeq mutant fitness 不可与普通 insertion counts 混同 |
| 15 | [BV-BRC/p3_tnseq](https://github.com/BV-BRC/p3_tnseq) | `service-scripts/App-TnSeq.pl` | MIT wrapper；底层分析工具另算 |
| 16 | [pseudogene/vibrio-tnseq](https://github.com/pseudogene/vibrio-tnseq) | `docker/run_pipeline.pl` | GPLv3；*Vibrio* study-specific |
| 17 | [bgoodlab/adaptation_tnseq](https://github.com/bgoodlab/adaptation_tnseq) | `scripts/generate_fitness_profile_data.py` | root license 未发现；prepared lineage/time-series study data，不是通用 raw-read pipeline |
| 18 | [amzonmeeee/phage_TnSeq_visualization](https://github.com/amzonmeeee/phage_TnSeq_visualization) | `src/phage_tnseq_viz/essentiality.py` | MIT；phage domain, 不计作细菌证据 |
| 19 | [Franciscohss/TN-SEQ-ANALYZER](https://github.com/Franciscohss/TN-SEQ-ANALYZER) | `src/server/analysis/stats_models.py` | MIT；web app 的 essentiality statistics |
| 20 | [jsa-aerial/aerobio](https://github.com/jsa-aerial/aerobio) | `Jobs/tradis-compare.clj` | MIT workflow wrapper；调用另一个 fitness tool |

## 正反合 / 架构影响

**正题：** TRANSIT、Bio-Tradis、TSAS、FiTnEss、RB-TnSeq/BarSeq 等代码体现了不同插入突变/reads 数据形态及不同的 essentiality/fitness 统计工作流。

**反题：** raw-count pipeline、已存在 fitness calls 的聚合、guide-library design、BarSeq、物种专用 study 分析并非可互换接口。Saturated library、条件、strain、时间点、统计假设都会改变 essentiality/fitness 解释；软件存在不等于目标菌数据也满足假设。

**合题 / 架构改变：** PACE/H1 输入要求时间点与条件 provenance；少点/秩亏返回 `not_fitted`。EIB202 私有组学缺失时 H1 unknown，不以这些仓库的跨物种数据伪造目标菌 gold。

**可证伪检测器：** 目标菌、预注册 assay 的独立 held-out time course；报告按条件的 fitness association、top-k hit rate 和不确定区间。若只在文献/同源数据有效而在 blinded target assay 无效，则拒绝跨物种 H1 迁移。

## 安全/方法边界

本轮只检查软件如何读取并计算公开功能基因组数据，未执行突变设计、培养或实验操作。Round 5 文献随机全文复核的另一个子任务被平台安全过滤阻断；本报告不通过改写查询绕过。针对疫苗效力的实验结论保持未验证。
