# GCDTO 迭代 10 — 去重、同源分组与 benchmark 泄漏代码核验

**已人工检查公开仓库数：20/20（仅本轮代码配额）。** 2026-10-07 对每个仓库读取了 README、相关源码路径及 license/status。没有在 VaccineGPT 数据上运行聚类或计算泄漏率。

| Repository | 作用 | 已检查代码路径 | License 状态 |
|---|---|---|---|
| [MMseqs2](https://github.com/soedinglab/MMseqs2) | 序列搜索/聚类 | `src/workflow/Cluster.cpp` | `LICENSE` 存在 |
| [CD-HIT](https://github.com/weizhongli/cdhit) | identity clustering | `cdhit.c++` | `license.txt`，GPLv2 |
| [Foldseek](https://github.com/steineggerlab/foldseek) | 结构相似度聚类；不可冒充序列 identity | `src/FoldseekBase.cpp` | `LICENSE.txt`、GPL3 文件存在 |
| [NCBI C++ Toolkit](https://github.com/ncbi/ncbi-cxx-toolkit-public) | BLAST search/alignment | `src/algo/blast/core/blast_engine.c` | 源文件有 Public Domain notice；根 license 文件未确认 |
| [SeqKit](https://github.com/shenwei356/seqkit) | FASTA/Q utilities、`rmdup` | `seqkit/cmd/rmdup.go` | MIT |
| [Seqtk](https://github.com/lh3/seqtk) | FASTA/Q utilities | `seqtk.c` | `LICENSE` 存在 |
| [VSEARCH](https://github.com/torognes/vsearch) | Dereplication/identity clustering | `src/commands/cluster_fast.cpp`, `derep_fulllength.cpp` | GPLv3/BSD 2-Clause dual terms |
| [DIAMOND](https://github.com/bbuchfink/diamond) | Protein alignment/search | `src/cluster/data.cpp` | GPL-3.0-or-later |
| [Biopython](https://github.com/biopython/biopython) | FASTA parsing | `Bio/SeqIO/__init__.py` | Biopython License Agreement/BSD notice |
| [scikit-bio](https://github.com/scikit-bio/scikit-bio) | Sequence representation/distance | `skbio/sequence/distance.py` | Modified BSD notice |
| [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) | Protein variant benchmark | `proteingym/merge.py` | `LICENSE` 存在 |
| [TAPE](https://github.com/songlab-cal/tape) | Protein representation benchmarks | `tape/datasets.py` | `LICENSE` 存在 |
| [FLIP](https://github.com/J-SNACKKB/FLIP) | Protein-engineering benchmarks/split materials | `collect_splits/helpers.py` | Root `LICENSE` 存在 |
| [ProteinNet](https://github.com/aqlaboratory/proteinnet) | Protein structure benchmark parser | `code/text_parser.py` | `LICENSE` 存在 |
| [ESM](https://github.com/facebookresearch/esm) | Protein language model/sequence handling | `esm/data.py` | `LICENSE` 存在 |
| [ProtTrans](https://github.com/Rostlab/ProtTrans) | Protein LMs/benchmarks | `Benchmark/ProtBert.ipynb` | `LICENSE.md` 存在 |
| [OpenFold](https://github.com/aqlaboratory/openfold) | Structure data pipeline | `openfold/data/data_pipeline.py` | `LICENSE` 存在 |
| [CombinGym](https://github.com/sitonglab/CombinGym) | Protein-fitness benchmarks | `CombinGym/FLIP/collect_splits/` | Root and nested FLIP `LICENSE` present |
| [PEER Benchmark](https://github.com/DeepGraphLearning/PEER_Benchmark) | Protein benchmark/evaluation | `peer/engine.py` | `LICENSE` 存在 |
| [MAVENN](https://github.com/jbkinney/mavenn) | MAVE data/model validation | `mavenn/src/mavedb.py`, `mavenn/src/validate.py` | `LICENSE` 存在 |

## 正反合与项目改动

**正题：** MMseqs2/CD-HIT/VSEARCH expose explicit similarity clustering workflows; SeqKit implements exact-record filtering; parsers/benchmark repositories show practical ingestion and split-support code.

**反题：** exact duplicate and homolog clustering are different; structural similarity (Foldseek) is not identical to sequence identity; BLAST performs search/alignment but does not, by itself, define a cluster-disjoint benchmark; a repository's existence does not prove its dataset split is leakage-free.

**合题 / code change:** split grouping and label provenance stay separate in VaccineGPT. The M1/M2 training entry points require MMseqs2 `90%` cluster IDs and refuse missing groups; the exact-ID/sequence audit does not claim homolog control. A future real run must store identity definition, coverage mode, sensitivity, representative rule, cluster ID, split assignment, and cross-split nearest-hit audit.

**Falsifiable test:** execute MMseqs2 clustering on the complete data, assign groups before preprocessing/training, then measure nearest homologs across train/validation/test. Compare row-random vs cluster-held-out metrics and reject leakage-safe claims if any same-cluster group crosses splits.

**Scope boundary:** no exact duplicate/homology counts or leakage rate were computed for the 100GB processed corpus; repository and dataset licensing remain separate. `SeqKit rmdup` uses hash-keyed state; a robust collision-safe exact deduplicator must compare normalized sequence bytes after hash collision.
