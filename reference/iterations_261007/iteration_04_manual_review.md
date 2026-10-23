# GCDTO 迭代 04 — Reverse vaccinology 与保护终点

**问题：** 预测的 antigenicity/epitope 与实测抗体结合、功能免疫和保护效力的证据等级应如何分开？

**配额状态：** round 04 下载 200/200 OA fullTextXML。AI 子代理辅助审阅随机样本 10 篇摘要及抽取的 Methods/Results/Discussion；大多数节选 ≤1,800 字符，不能把节选未显示内容说成全文未报告，也不代表人工逐篇深读。完整参考集在 `..\data\research_campaign_261007\rounds\round_04_references.jsonl`。

## 抽样全文证据

| Work | 实际终点/阅读定位 | 迁移边界 |
|---|---|---|
| *S. suis* epitope study, DOI `10.3390/vetsci10060383`, PMC10301797 | Abstract、§2、§3、§4、§5：计算性 epitope、population coverage、TLR4 docking/稳定性 | 未见实测 binding、immunogenicity、challenge/protection；计算性 cross-protection 不等于观察到的保护 |
| *R. equi* reverse vaccinology, DOI `10.1016/j.heliyon.2023.e18623`, PMC10413060 | Abstract、Materials/Methods、Results、Discussion、Conclusion：筛 12 candidates/29 epitopes，作者明示需实验验证 | PDB structure 的实测性不等于这些候选已被保护试验验证；16 个采样 proteome 的 conservation 不能代表广泛保护 |
| Viral carp epitope comparator, DOI `10.1038/s41598-024-61074-7`, PMC11116410 | Abstract/Methods/Results/Discussion：计算 epitope docking/simulation | 病毒，不是细菌；作者称尚需体内/田间试验 |
| *V. cholerae* O1, DOI `10.1038/s41598-025-90598-9`, PMC11958690 | Abstract/Methods/Results/Discussion：单 O1 genome 从约 19,155 ORFs 计算筛选两候选 | 计算 antigenicity/docking，不是测量 binding/免疫/挑战保护；单菌株不证明跨 O1/O139 或跨宿主 |
| *Leishmania* VLP study, DOI `10.1128/msphere.00097-25`, PMC12108087 | Abstract 报小鼠 challenge 与 parasite burden，含 75% reduction；Discussion 指明为部分保护动物血清来源的 candidate | 这是 in-vivo parasite-burden evidence，但不是 bacterial vaccine evidence；challenge 结果在本 review packet 中为摘要级 |
| Viral vaccine design, DOI `10.1038/s41598-025-28522-4`, PMC12749620 | Abstract/Methods/Results/Discussion/Conclusion：计算性 epitope/docking/MD | 计算预测不是实测免疫或保护；病毒外推不构成细菌证据 |
| Meningococcal monoclonal antibody study, DOI `10.3389/fimmu.2018.01621`, PMC6055031 | Abstract/Methods/Results/Discussion：实测 binding 与 serum bactericidal activity；结果指出部分 binding IgG 没有 bactericidal activity | 实测功能活性高于纯结合证据，但自然感染、人 monoclonal、非疫苗challenge；target protein 当时未解析 |
| Aquaculture pathogen reverse vaccinology, DOI `10.3390/vaccines10030473`, PMC8954842 | Abstract、§2–§5：多 pathogen 公共抗原和 predicted MHC-II docking，结论需实验验证 | 其它 fish-pathogen 的既往 protection citations 是异源先验，不是本研究当前 antigen 的试验结果 |
| *B. mallei* computational vaccine, DOI `10.3390/vaccines10101580`, PMC9607922 | Abstract、§2–§5：预测 Hcp/PilQ/epitopes 与 docking/MD，Conclusion 称需实验分析 | “proven antigenicity”在本研究是 computational，不是 protection study |
| Viral computational vaccine, DOI `10.3389/fimmu.2022.1022159`, PMC9692015 | Abstract、Methods、Results、Discussion、Conclusion：计算 antigen/epitope、docking/simulation，建议实验验证 | 病毒 comparator，不能作为 bacterial protection 的阳性实测证据 |

## 正题与反题

**正题：** 公开 reverse-vaccinology 软件与研究能压缩候选空间；跨 pathogen conserved antigens 可作为待测候选，计算 epitope、序列保守性和 topology/structure score 可作为先验。

**反题：**

1. Predicted binding、antigenicity、免疫模拟不等于实测 immune response，更不等于 challenge protection。
2. Meningococcal 研究中 observed antibody binding 未能保证 SBA，展示 binding 与功能终点的非等价性。
3. 某些小鼠/鱼/病毒 study 的体内或体外结果，仅对原物种、原 pathogen、原构建体和原试验条件直接；不自动迁移到 EIB202 或其它宿主。

## 合题 / AEP 改动

证据链显式区分：**计算预测 → native antigen 实测 recognition → 功能活性（SBA/OPA 等）→ 目标宿主 immunogenicity → 目标菌 challenge protection → 异源株/现场复现**。AEP 的 evidence grade 必须记录观察到的最高直接终点和它与 target strain/host 的直接性；高 embedding/docking 分数不得抵消“无功能/无保护试验”。

## 可证伪目标试验

选一候选（例如某报道的 MVHP6）做目标鱼/目标菌挑战实验：随机分配疫苗+佐剂、佐剂-only、irrelevant-protein controls；盲法评估疾病/存活/血液或组织菌量为预注册主要终点，native-cell recognition 和 SBA/OPA 作次要终点。若目标 endpoint 不优于佐剂对照，或只在同型而不在预设异型株有效，则 cross-protection 主张被否定，不因计算 antigenicity 改判。
