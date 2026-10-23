# GCDTO 迭代 06 — Annotation、ORF boundary 与 provenance

**配额：** round 06 OA fullTextXML 200/200；AI 子代理辅助审阅随机 10 篇 abstract 与选定 Methods/Results/Discussion excerpts。Direct=片段支持所述主张，不代表人工逐篇通读或独立重现整篇论文；abstract-only 项单列。

## 抽样证据

| Work | 可直接支持的观察与位置 | 限制 |
|---|---|---|
| Black rhino draft lift-over, DOI `10.1186/s12863-024-01214-0`, PMC10916078 | Data description：从 GCA_013634535.1 lift-over >99.9% gene features | feature recovery 不是 ORF boundary accuracy；834 contigs assembly仍 fragmented |
| Heavy-metal resistance promoter prediction, DOI `10.1177/11779322241299905`, PMC11587186 | Methods：15 genes取1kb upstream，NNPP v2.2 cutoff 0.8预测TSS | computational预测，不是实测ORF边界；作者要求wet-lab确认 |
| SARS-CoV-2 metadata comparison, DOI `10.3390/microorganisms12112187`, PMC11596409 | Methods：MW750862在不同时点tree snapshot采集日期不同 | 单例显示metadata drift，不能推出频率或更广结论正确 |
| Bacterial AMR assembly panel, DOI `10.1128/spectrum.01947-25`, PMC12671191 | Methods/Results：>500 contigs或<90% completeness剔除；198个assembly满足QC | assembly-level QC不是CDS边界/pseudogene准确性；下游注释Methods截断 |
| Phage benchmark curation, DOI `10.1093/nar/gkaf1448`, PMC12774648 | Methods：Pharokka 1.7.4/1.7.5 + pyrodigal-gv calls CDS；后续结构同源/人工规则分别给功能标签 | function label curation不验证ORF；phage tool transfer边界 |
| Genome data paper, DOI `10.1038/s41597-025-06324-7`, PMC12775540 | Abstract报告 12,751 protein coding genes，11,305 functional annotation | counts为摘要级证据；节选未见注释方法/质量验证 |
| AquaaG, DOI `10.1016/j.mex.2026.103955`, PMC13200041 | Method details：记录assembly accession/species/status/N50，下载GFF与protein FASTA | 作者指出assembly quality依赖；BUSCO可能遗漏novel genes，注释限EggNOG |
| *B. cereus* BT-C2.4 data, DOI `10.1016/j.dib.2026.113083`, PMC13396940 | Abstract：NCBI PGAP 注释 | annotation版本/验证为摘要级；单isolate draft/multi-contig的代表性有限 |
| SARS-CoV-2 Prokka comparison, DOI `10.3390/v13122426`, PMC8706859 | Comparative analysis/Appendix：Prokka v1.14.5与2020-09-16 UniProt pre-release；reference protein membership/length比对报告缺失/截短 | virus-specific；reference比对非独立实验truth |
| Parasite BRAKER RNA-seq gene models, DOI `10.3390/pathogens11111273`, PMC9694666 | Methods/Results：记录WormBase assembly/annotation版本，RNA-seq支持BRAKER，GffRead 0.12.1提取complete ORFs | parasite-eukaryote证据；未看到pseudogene benchmark，不直接迁移到细菌 |

## 正题/反题/合题

**正题：** assembly accession/version、工具/数据库release与checksum、RNA/protein orthogonal evidence可提升注释结果的可追溯性。  
**反题：** 注释计数、lift-over率、BUSCO或工具名称不是逐ORF准确率；异种/病毒工具证据不能当目标细菌金标签；本抽样未发现直接验证细菌 pseudogene classification 的证据。  
**合题 / 数据合同：** 记录 assembly accession.version/checksum 与 annotation provenance 分开；feature级保留坐标、链、frame、translation table、complete/partial、contig-edge、pseudogene status、disruption reason 与 evidence source。推断、转移和实验支持绝不混级。

## 可证伪QC试验

对盲法 curated 的 100 个完整细菌 CDS + 20 个validated pseudogenes，锁定输入assembly和annotation release；通过标准是≥95/100完整 CDS 的起止坐标完全匹配、contig-edge 必标 partial、20个 pseudogene 全有状态和disruption reason且不得作为 intact translation 输出。任一 provenance 缺失或阈值未达即失败。
