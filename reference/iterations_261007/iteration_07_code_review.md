# GCDTO 迭代 07 — Biological graph/GAT/PPI code 核验

**已检查 20/20 个 public repositories**，包括 GAT/GATv2 libraries 与 PPI/structure/regulatory graph examples。未找到经过核实的 operon-specific GNN implementation；operon graph 与相关 GNN task 的迁移仍是 inference。

| Repository | 实现路径 | License status |
|---|---|---|
| [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | `torch_geometric/nn/conv/gatv2_conv.py` | MIT |
| [dmlc/dgl](https://github.com/dmlc/dgl) | `python/dgl/nn/pytorch/conv/gatv2conv.py` | Apache-2.0 |
| [tech-srl/how_attentive_are_gats](https://github.com/tech-srl/how_attentive_are_gats) | `gatv2_conv_PyG.py`, `gatv2_conv_DGL.py`, `proteins/` | 根 license 未确认；nested path only |
| [PetarV-/GAT](https://github.com/PetarV-/GAT) | `models/gat.py`, `models/sp_gat.py` | root LICENSE present, type unverified |
| [gordicaleksa/pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT) | `models/definitions/GAT.py`, `training_script_ppi.py` | MIT |
| [karbalayghareh/GraphReg](https://github.com/karbalayghareh/GraphReg) | `train/gat_layer_v2.py`, `utils/hic_to_graph.py` | no root license found |
| [nathanyl/scMGATGRN](https://github.com/nathanyl/scMGATGRN) | `Code/scGNN.py`, `Code/main.py` | no root license found |
| [czi-ai/GREmLN](https://github.com/czi-ai/GREmLN) | `scGraphLLM/network.py`, `graph_op.py` | MIT |
| [BioinfoMachineLearning/GRNformer](https://github.com/BioinfoMachineLearning/GRNformer) | `src/models/grnformer/network.py`, `model.py` | MIT |
| [nrr-90/MultiCausGRN](https://github.com/nrr-90/MultiCausGRN) | `Model/Code/scGNN.py`, `main.py` | no root license found |
| [ZJUDataIntelligence/GNN-PPI](https://github.com/ZJUDataIntelligence/GNN-PPI) | `layers/att_layers.py` | root license present; type not classified |
| [ohuelab/SpatialPPIv2](https://github.com/ohuelab/SpatialPPIv2) | `utils/model.py`, `utils/dataset.py` | Apache-2.0 |
| [podgorskip/protein-biological-function-prediction](https://github.com/podgorskip/protein-biological-function-prediction) | BioGRID notebook compares GAT/GCN/GraphSAGE | no root license found |
| [Sazan-Mahbub/EGRET](https://github.com/Sazan-Mahbub/EGRET) | `EGRET/models/EdgeAggregatedGAT_attention_visual.py` | MIT |
| [flatironinstitute/DeepFRI](https://github.com/flatironinstitute/DeepFRI) | `deepfrier/DeepFRI.py`, `layers.py` | BSD-3-Clause |
| [yourh/DeepGraphGO](https://github.com/yourh/DeepGraphGO) | `deepgraphgo/networks.py`, `models.py` | no root license found |
| [zwh0511/GRDGNN](https://github.com/zwh0511/GRDGNN) | `code/Transductive/`, `code/Inductive/` | `LICENCE` present, type unverified |
| [juexinwang/scGNN](https://github.com/juexinwang/scGNN) | `graph_function.py`, `scGNN.py` | MIT |
| [EmmaDPaul/GRN-inference-using-GNN](https://github.com/EmmaDPaul/GRN-inference-using-GNN) | preprocessing and inference notebooks | no root license found |
| [Drax929/Automated-Pathway-Reconstruction-using-Attention-Guided-GNN](https://github.com/Drax929/Automated-Pathway-Reconstruction-using-Attention-Guided-GNN) | `protein_gnn_final.ipynb` | no root license found |

## 正反合

**正题：** PyG/DGL implement GATv2; GraphReg/GRDGNN/GRNformer show graph/attention use on chromatin/regulatory tasks; PPI and protein-structure graph repositories show sequence/contact graphs can support their task-specific benchmark.

**反题：** PPI prediction, residue interface quality, GO function, gene-regulatory reconstruction and cell graphs have distinct nodes, edge semantics and labels. A GAT coefficient is a message weight, not a causal effect. No operon-specific GNN validation was verified here.

**合题 / implementation:** GCE edges remain source/evidence typed; PPI/STRING predicted edges E3; no attention-to-causality claims. Operon adjacency is an annotation/position prior, not a learned validated operon graph.

**Falsifiable detector:** target-genome/cluster-heldout GCE/no-GCE, per-edge-type ablation and degree-preserving rewiring; require pre-registered gain and stable effect across species before claiming graph utility.
