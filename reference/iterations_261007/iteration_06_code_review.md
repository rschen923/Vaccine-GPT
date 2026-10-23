# GCDTO 迭代 06 — bacterial genome annotation / ORF code 核验

**已检查 20/20 公共 GitHub 仓库**，含 README、相关源码路径和 license file/status。组装、ORF 预测、功能注释、耐药/毒力筛查、workflow wrappers 的功能边界和数据许可分开核验。

| Repository | 作用/路径 | License 与边界 |
|---|---|---|
| [Prodigal](https://github.com/hyattpd/Prodigal) | ORF/coding gene prediction; `gene.c` | GPLv3 |
| [Bakta](https://github.com/oschwengers/bakta) | genome/MAG/plasmid annotation; `bakta/proteins.py`, `bakta/main.py` | GPLv3; 数据库另审 |
| [Prokka](https://github.com/tseemann/prokka) | prokaryotic annotation orchestrator; `bin/prokka` | GPLv3 |
| [NCBI PGAP](https://github.com/ncbi/pgap) | genome annotation workflow/pass 4; `bacterial_annot/wf_bacterial_annot_pass4.cwl` | LICENSE distinguishes government/third-party and GeneMarkS restrictions |
| [eggNOG-mapper](https://github.com/eggnogdb/eggnog-mapper) | orthology-supported function annotation; `emapper.py` | AGPLv3 code; database separate |
| [DFAST core](https://github.com/nigyta/dfast_core) | DDBJ prokaryote annotation; `dfast` | root license not found; unknown |
| [Pyrodigal](https://github.com/althonos/pyrodigal) | Python/Cython Prodigal bindings; `src/pyrodigal/lib.pyx` | GPLv3 |
| [KofamScan](https://github.com/takaram/kofam_scan) | KOfam HMM function annotation; `lib/kofam_scan/cli.rb` | MIT code; HMM data separate |
| [InterProScan](https://github.com/ebi-pf-team/interproscan) | protein functional classification; `core/` | Apache-2.0 code; member DBs separate |
| [antiSMASH](https://github.com/antismash/antismash) | biosynthetic gene-cluster discovery; `antismash/main.py` | AGPLv3 |
| [Pseudofinder](https://github.com/filip-husnik/pseudofinder) | pseudogene candidate detection; `modules/sleuth.py` | GPLv3 |
| [Barrnap](https://github.com/tseemann/barrnap) | microbial rRNA feature detection; `bin/barrnap` | GPLv3 + separate Rfam license |
| [ABRicate](https://github.com/tseemann/abricate) | database search for resistance/virulence genes; `bin/abricate` | GPLv2; database separately versioned/licensed |
| [RGI](https://github.com/arpcard/rgi) | CARD-based AMR gene/mutation detection; `app/Analyser.py` | CARD/ARO-specific terms; data/model terms distinct |
| [METABOLIC](https://github.com/AnantharamanLab/METABOLIC) | metabolic trait profiling; `METABOLIC-C.pl` | Source header GPLv3; no root LICENSE confirmed |
| [nf-core/bacass](https://github.com/nf-core/bacass) | bacterial assembly/annotation workflow; `main.nf`, local modules | MIT workflow; each tool retains its license |
| [nf-core/mag](https://github.com/nf-core/mag) | metagenome assembly/binning/annotation; `main.nf` | MIT workflow; tools/database separate |
| [Galaxy tools-iuc](https://github.com/galaxyproject/tools-iuc) | Prokka Galaxy wrapper; `tools/prokka/prokka.xml` | MIT wrapper; wrapper does not relicense Prokka |
| [anvi’o](https://github.com/merenlab/anvio) | microbial-genomics feature utilities; `anvio/utils.py` | GPLv3 |
| [NCBI AMRFinderPlus](https://github.com/ncbi/amr) | AMR/mutation, selected virulence factor detection; `amrfinder.cpp` | NCBI software/data public-domain notice |

## 正反合 / architecture impact

**正题：** Separate ORF callers, function-annotation systems, pseudogene detectors and broad workflows exist; some projects explicitly preserve database/tool versions and third-party license terms.

**反题：** ORF call ≠ correct protein; orthology/HMM/similarity label ≠ measured activity; annotation completeness, partial ends, pseudogenes, and database versions differ by pipeline. A single “annotation confidence” hides these error modes.

**合题 / implementation:** Keep ORF/sequence quality fields and source/version lineage separate from evidence level; preserve partial/pseudogene/unknown states and source identifiers; do not promote annotation to E1/E2. Pin tool and each underlying database independently. This directly motivated sequence-role separation and unique-sequence ID recovery.

**Falsifiable detector:** Stratify a random manually adjudicated annotation sample by pipeline, contig-edge/partial state, pseudogene status, and database version; report precision/recall and ID reconciliation. Reject annotation-derived gold labels if target-specific agreement falls below a predeclared threshold.
