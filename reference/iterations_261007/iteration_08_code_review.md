# GCDTO 迭代 08 — Calibration / conformal / uncertainty code 核验

**已检查 20/20 个公开 GitHub 仓库**，其中 19 个有可见 license file，1 个根目录未发现许可证。代码“实现了 conformal/calibration”不等于某 VaccineGPT 数据满足 exchangeability，也不等于生物学预测可靠。

| # | Repository | 已检查实现路径 | License 状态/限制 |
|---:|---|---|---|
| 1 | [scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | `mapie/classification.py` split conformal classifier | BSD-3-Clause |
| 2 | [henrikbostrom/crepes](https://github.com/henrikbostrom/crepes) | `src/crepes/base.py` predictors | BSD-style |
| 3 | [EFS-OpenSource/calibration-framework](https://github.com/EFS-OpenSource/calibration-framework) | `netcal/AbstractCalibration.py` | Apache-2.0 |
| 4 | [awslabs/fortuna](https://github.com/awslabs/fortuna) | `fortuna/conformal/classification/base.py` | Apache-2.0；README 标为 archived |
| 5 | [deel-ai/puncc](https://github.com/deel-ai/puncc) | `deel/puncc/classification.py` | MIT；README 明列 exchangeability/i.i.d 条件 |
| 6 | [donlnz/nonconformist](https://github.com/donlnz/nonconformist) | `nonconformist/cp.py` | MIT |
| 7 | [yromano/cqr](https://github.com/yromano/cqr) | `nonconformist/nc.py` CQR functions | MIT |
| 8 | [aangelopoulos/conformal_classification](https://github.com/aangelopoulos/conformal_classification) | `conformal.py` | MIT |
| 9 | [biolab/orange3-conformal](https://github.com/biolab/orange3-conformal) | `orangecontrib/conformal/classification.py` | GPL-3.0 |
| 10 | [HenDav/conformal_surv_general](https://github.com/HenDav/conformal_surv_general) | `calibration.py` | MIT |
| 11 | [ronboger/conformal-protein-retrieval](https://github.com/ronboger/conformal-protein-retrieval) | `protein_conformal/cli.py` Venn–Abers/threshold calibration | LICENSE Apache-2.0, README says MIT: conflict unresolved |
| 12 | [clarafy/conformal-for-design](https://github.com/clarafy/conformal-for-design) | `calibrate.py` covariate shift | MIT |
| 13 | [adamesalles/experiments-conformal-superres](https://github.com/adamesalles/experiments-conformal-superres) | `src/conformal.py` | MIT |
| 14 | [gpleiss/temperature_scaling](https://github.com/gpleiss/temperature_scaling) | `temperature_scaling.py` validation-NLL fitting | MIT |
| 15 | [uncertainty-toolbox/uncertainty-toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox) | `uncertainty_toolbox/metrics_calibration.py` | MIT |
| 16 | [google/uncertainty-baselines](https://github.com/google/uncertainty-baselines) | `baselines/cifar/utils.py` ECE metrics | Apache-2.0 |
| 17 | [chemprop/chemprop](https://github.com/chemprop/chemprop) | `chemprop/uncertainty/calibrator.py` | MIT |
| 18 | [laurapiro17/disease-gene-network-prediction](https://github.com/laurapiro17/disease-gene-network-prediction) | `src/conformal.py` threshold/candidate set | MIT |
| 19 | [layer6ai-labs/hitl-conformal-prediction](https://github.com/layer6ai-labs/hitl-conformal-prediction) | `raps/conformal.py` | MIT |
| 20 | [octadion/med-conformal](https://github.com/octadion/med-conformal) | `conformal.py` | no root LICENSE found; reuse permission unknown |

## 正反合 / architecture impact

**正题：** General-purpose libraries support split conformal classifiers/regressors, calibration diagnostics, CQR, survival conformal methods, and probability scaling. Protein-retrieval and biomolecular-design repositories show task-specific applications.

**反题：** Coverage depends on exchangeability; homology clusters, repeated assays, target-domain shift, and label error violate that premise. Conformal interval coverage ≠ calibrated posterior probability; calibrated probabilities ≠ correct ranking; neither proves protection. The protein retrieval repo also has a license conflict; Fortuna is archived.

**合题 / implementation:** Keep model calibration, uncertainty intervals, ranking and biological evidence separate. Require a representative, group-disjoint calibration set and return `[0,1]/unavailable` when absent; do not let wide/narrow intervals auto-promote candidates.

**Falsifiable detector:** On untouched target-strain and homology-group heldout sets, measure empirical coverage, interval width, subgroup coverage and calibration. Fail the gate if the pre-registered 90% interval coverage misses its tolerance or calibration degrades across species/assays.
