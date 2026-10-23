# GCDTO 迭代 10 — Exact duplicates / homology / benchmark leakage

**配额：** round 10 下载 200/200 OA fullTextXML；AI 子代理辅助核查随机抽样 10 篇的摘要与选定 sections。这不是人工逐篇通读。该 packet 没有 VaccineGPT 的 raw sequences/split manifests/graph，不能计算该项目的重复率或泄漏率。

## 抽样证据

| Work | Sections 与观察 | 限制 |
|---|---|---|
| Octopus cathepsins, `10.1242/bio.061778`, PMC12032550 | Methods/Results称 117 reference sequences + transcriptome/genome；经过 BLASTX/过滤保留6,467 contigs“without duplicates” | 归一化/重复相等条件未给；不是 train/test split审计；功能来自同源/树推断，未实测 |
| Goose circovirus isolates, `10.1016/j.psj.2025.105826`, PMC12466204 | Rep/Cap 对比报告某些分离株蛋白100% aa identity | 等长蛋白对可为 exact duplicate候选，但excerpt无alignment coverage；10株疫情监测非benchmark |
| Phage GAF proteins, `10.1016/j.csbj.2025.09.020`, PMC12481020 | Results/Discussion用Foldseek/DALI找低序列相似结构候选 | 作者讨论可能收敛演化；结构相似≠序列同源/功能确认 |
| Protein alignment embeddings, `10.1038/s41598-025-23319-x`, PMC12612178 | PISCES 19,599 pairs, reported max identity 30%；评估 embedding 对 TM score相关 | 30%是该benchmark阈值，不是普遍安全阈值，不能当 GCDTO leakage rate |
| Carrot HSP70 family, `10.1186/s12870-026-08102-y`, PMC12892451 | 52 HSP70 family members、segmental duplication | paralog/family成员是生物同源，不自动代表重复记录或泄漏 |
| CAP protein-nucleic-acid affinity, `10.1088/2632-2153/ae29bc`, PMC13001652 | S186/S142/S322 datasets；文中提到多个研究使用重叠数据 | 跨论文复用本身不等于train/test leakage；excerpt没提供split membership |
| TRACE-DDI, `10.34133/csbj.0057`, PMC13176605 | 明确将 pair-level stratified CV 称 transductive；1,706 drugs/191,417 pairs | 同一 drug 跨 fold并非 unseen-drug测试；excerpt未证明 graph label-edge leakage发生 |
| scGPT fine-tuning/BO, `10.1093/bioinformatics/btag374`, PMC13360271 | Methods/Results比较多dataset/seeds与optimization | 不是蛋白序列同源benchmark；split边界 excerpt不完整 |
| RIBEX, `10.1093/bioinformatics/btag471`, PMC13501326 | human PLM + STRING PPI context预测 RNA-binding propensity | graph context是否泄漏target-associated information未证；annotation和RIC endpoints不同 |
| ESpma interface prediction, `10.1002/pro.70790`, PMC13584886 | Methods用random 80:20 train/val及单独162-interface test；说明EPPIC在primary test数据开发，作一致性比较 | EPPIC对比不是独立benchmark；exact redundancy threshold及structure-family split信息不完整 |

## 正题/反题

**正题：** exact duplicate、同源簇、transductive node/pair split 和 benchmark reuse 是不同审计对象；protein benchmarks会使用 identity/group thresholds；同源与重复也可能是有意研究对象，不能简单删掉所有related entries。

**反题：** 目前这些paper的数字/identity阈值不能移植为 GCDTO 数据的泄漏率或 universal threshold。数据库/图source相关性可造成潜在泄漏，但未查GCDTO split/edge provenance就不能声称已经污染或已无泄漏。

## 合题 / 数据契约

保持：

1. 原始 record accession/文件位置/sequence checksum；
2. exact-normalized sequence group；
3. sequence homology cluster（工具、版本、identity/coverage/sensitivity参数）；
4. genome/locus/operon group；
5. train/validation/calibration/test group IDs；
6. graph edge source and timestamp、target label source、冲突标志。

Exact hash查重不可替代MMseqs2聚类；BLAST是search/alignment工具，不是自动的cluster-disjoint split。Structural similarity另立字段，不能当sequence identity。

## 可证伪审计

对冻结数据运行 exact duplicate、预注册 MMseqs2 ≥90% identity/coverage聚类及每组分割归属统计；计算跨fold nearest hits与label/source/edge交集。若同一预注册 group跨 train/test，group-heldout gate fail。只有实际运行结果才可报告具体泄漏率；本轮未在 GCDTO 真实 100GB corpus完成 homology scan。
