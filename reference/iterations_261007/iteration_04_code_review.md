# GCDTO 迭代 04 — Reverse vaccinology / bacterial vaccine code 核验

**已检查 20/20 公共仓库：**6 个直接疫苗/抗原/epitope 管线，14 个上游或邻接工具。许可证未知的条目保留 unknown；README/实现代码仅证明软件意图/能力，不证明保护效力。

| Repository | README/implementation source | License/status |
|---|---|---|
| [VIOLINet/Vaxign2-django](https://github.com/VIOLINet/Vaxign2-django) | `README.md`, `vaxign/views/analysis.py` | 根 LICENSE 未发现；未知 |
| [VIOLINet/Vaxign-ML-docker](https://github.com/VIOLINet/Vaxign-ML-docker) | `README.md`, `VaxignML.py` | LICENSE 存在，类型未从文件核实 |
| [nerve-bio/NERVE](https://github.com/nerve-bio/NERVE) | `README.md`, `code/NERVE.py` | LICENSE 存在，类型未核实 |
| [SravanKrishnaR/VaxiFlow](https://github.com/SravanKrishnaR/VaxiFlow) | `README.md`, `main.nf` | LICENSE 存在，类型未核实 |
| [RuijinHospitalVNAR/AntigenFinder](https://github.com/RuijinHospitalVNAR/AntigenFinder) | `BacterialAntigenFinder-2/README.md`, `src/aggregator/candidate_ranker.py`, `src/predictors/vaxignml_wrapper.py` | 检视的根目录无 license，未知 |
| [bioinformatics-007/EpitopeFinder](https://github.com/bioinformatics-007/EpitopeFinder) | `README.md`, `core_pipeline.py` | README 声明 MIT，LICENSE 文件存在 |
| [oschwengers/bakta](https://github.com/oschwengers/bakta) | `README.md`, `bakta/main.py` | GPLv3 |
| [tseemann/prokka](https://github.com/tseemann/prokka) | `README.md`, `bin/prokka` | GPLv3 |
| [tseemann/abricate](https://github.com/tseemann/abricate) | `README.md`, `bin/abricate` | GPLv2 |
| [ncbi/amr](https://github.com/ncbi/amr) | `README.md`, `amrfinder.cpp` | NCBI Public Domain Notice |
| [gtonkinhill/panaroo](https://github.com/gtonkinhill/panaroo) | `README.md`, `panaroo-runner.py` | LICENSE 存在，类型未核实 |
| [openvax/mhcflurry](https://github.com/openvax/mhcflurry) | `README.md`, `mhcflurry/predict_command.py` | LICENSE 存在，类型未核实 |
| [dmnfarrell/epitopepredict](https://github.com/dmnfarrell/epitopepredict) | `README.md`, `epitopepredict/base.py` | GPLv3 |
| [Superzchen/iFeature](https://github.com/Superzchen/iFeature) | `README.md`, `iFeature.py` | 根 LICENSE 未见，未知 |
| [flatironinstitute/DeepFRI](https://github.com/flatironinstitute/DeepFRI) | `README.md`, `predict.py` | LICENSE 存在，类型未核实 |
| [dansondergaard/tmhmm.py](https://github.com/dansondergaard/tmhmm.py) | `README.md`, `tmhmm/api.py` | LICENSE.md 存在，类型未核实 |
| [biopython/biopython](https://github.com/biopython/biopython) | `README.rst`, `Bio/SeqIO/__init__.py` | Biopython License Agreement/BSD notices |
| [sanger-pathogens/Roary](https://github.com/sanger-pathogens/Roary) | `README.md`, `bin/roary` | GPLv3；README 标记 unmaintained |
| [eggnogdb/eggnog-mapper](https://github.com/eggnogdb/eggnog-mapper) | `README.md`, `emapper.py` | `LICENSE.txt` 存在，类型未核实 |
| [sanger-pathogens/ariba](https://github.com/sanger-pathogens/ariba) | `README.md`, `ariba/assembly.py` | GPLv3；README 标记 unmaintained |

## 正反合与实现增量

**正题：** Vaxign-ML、Vaxign2、NERVE、VaxiFlow、AntigenFinder、EpitopeFinder 提供不同 proteome screening、机器学习、表位/候选排序代码；Bakta/Prokka/Panaroo/MHCflurry 分别支持注释、pan-genome 或表位呈递接口。

**反题：** 预测 pipeline 输出仍是计算候选。结构定位、sequence motif、MHC binding、GO function 都不是实验保护；部分仓库许可未知，部分 VaxiFlow/旧管线数据库、外部模型条款未核。

**合题 / 项目影响：** H2、H3、M3-S 与 upstream annotation 是可插拔证据层；其输出保留 E3/E4，不自动升格 E1/E2。呈递/免疫原性/保护三层不得合并。许可证 manifest 分列代码、权重与数据集。

**可证伪检测器：** 目标菌与目标宿主中，以盲法 challenge/表面表达 assay 测 top-k hits，对比 annotation-only 与 pipeline-ranked candidates；若保护/表达 hit-rate 无显著提升，撤销该排序器临床/疫苗意义主张。

