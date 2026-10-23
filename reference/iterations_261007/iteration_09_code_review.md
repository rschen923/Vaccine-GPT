# GCDTO 迭代 09 — Pareto / multiobjective / active learning code 核验

**本轮检查 22 个公共仓库**（目标20，配额达成）。含 multiobjective/BO/active-learning 库和生物应用代码。它们能执行优化机制，不提供 VaccineGPT 的生物学目标或验证。

| # | Repository | 实现路径 | License |
|---:|---|---|---|
| 1 | [anyoptimization/pymoo](https://github.com/anyoptimization/pymoo) | `pymoo/algorithms/moo/nsga2.py` | license present |
| 2 | [meta-pytorch/botorch](https://github.com/meta-pytorch/botorch) | `botorch/acquisition/multi_objective/monte_carlo.py` | license present |
| 3 | [DEAP/deap](https://github.com/DEAP/deap) | `deap/tools/emo.py` NSGA-II/III | LICENSE.txt |
| 4 | [jMetal/jMetalPy](https://github.com/jMetal/jMetalPy) | `src/jmetal/algorithm/multiobjective/nsgaii.py` | LICENSE |
| 5 | [Project-Platypus/Platypus](https://github.com/Project-Platypus/Platypus) | `platypus/algorithms.py` | LICENSE |
| 6 | [multi-objective/moocore](https://github.com/multi-objective/moocore) | `python/src/moocore/__init__.py` hypervolume/IGD | LICENSE |
| 7 | [optuna/optuna](https://github.com/optuna/optuna) | `optuna/samplers/nsgaii/_sampler.py` | LICENSE |
| 8 | [facebook/Ax](https://github.com/facebook/Ax) | `ax/generators/torch/botorch_moo_utils.py` | LICENSE |
| 9 | [secondmind-labs/trieste](https://github.com/secondmind-labs/trieste) | `trieste/acquisition/function/multi_objective.py` EHVI | LICENSE |
| 10 | [facebookresearch/nevergrad](https://github.com/facebookresearch/nevergrad) | `nevergrad/optimization/multiobjective/nsga2.py` | LICENSE |
| 11 | [EmuKit/emukit](https://github.com/EmuKit/emukit) | `emukit/experimental_design/experimental_design_loop.py` | LICENSE |
| 12 | [dragonfly/dragonfly](https://github.com/dragonfly/dragonfly) | `dragonfly/opt/multiobjective_gp_bandit.py` | LICENSE.txt |
| 13 | [scikit-optimize/scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) | `skopt/optimizer/optimizer.py` | LICENSE; general single-objective SMBO |
| 14 | [SheffieldML/GPyOpt](https://github.com/SheffieldML/GPyOpt) | `GPyOpt/methods/bayesian_optimization.py` | LICENSE.txt; single-objective |
| 15 | [modAL-python/modAL](https://github.com/modAL-python/modAL) | `modAL/uncertainty.py` | LICENSE |
| 16 | [ntucllab/libact](https://github.com/ntucllab/libact) | `libact/query_strategies/uncertainty_sampling.py` | LICENSE |
| 17 | [baal-org/baal](https://github.com/baal-org/baal) | `baal/active/heuristics/heuristics.py` | LICENSE |
| 18 | [deepchem/deepchem](https://github.com/deepchem/deepchem) | `deepchem/models/graph_models.py` | LICENSE; scientific ML substrate |
| 19 | [experimental-design/bofire](https://github.com/experimental-design/bofire) | `bofire/strategies/predictives/mobo.py` | LICENSE |
| 20 | [emdgroup/baybe](https://github.com/emdgroup/baybe) | `baybe/acquisition/acqfs.py` | LICENSE |
| 21 | [ErikHartman/bopep](https://github.com/ErikHartman/bopep) | `bopep/search/peptidome_search.py` | LICENSE |
| 22 | [COLA-Laboratory/GraphFLA](https://github.com/COLA-Laboratory/GraphFLA) | `graphfla/landscape/protein.py` | LICENSE |

## 正反合

**正题：** NSGA-II/III, hypervolume/EHVI, Pareto metrics, Bayesian optimization and uncertainty acquisition have source implementations; BoPep and GraphFLA are sequence/fitness landscape analogues.

**反题：** A Pareto front is not a unique winner. Objective normalization, constraint choice, reference points, acquisition function, and wet-lab measurement uncertainty are decision assumptions. Generic libraries do not define vaccine-valid objectives; protein fitness landscapes do not establish protection.

**合题 / implementation:** Use the Pareto set to expose Att/Ret trade-offs; apply predeclared H4/safety gates; use scalar U only within tiers; ACQ is experimental prioritization only when its calibration and assay inputs exist. Human decision criteria stay separate and explicit.

**Falsifiable test:** Compare uncertainty-aware MO acquisition, random selection, and preregistered single-objective baseline at fixed assay budget; evaluate experimentally measured feasible Pareto hypervolume on scaffold/family-heldout data. If no uplift or membership unstable under assay replicate noise, reject active-learning benefit.
