# Vaccine-GPT architecture audit and optimization

## Findings addressed in code

1. The authoritative five-layer path now validates batch dimensions, square
   PPI adjacency, node features, and the required feature widths.
2. `node_x` and `PPI_adj` now affect the shared state through a normalized
   graph message, rather than being required-but-unused fields.
3. `S_adj`, `t4_feat`, `N_star`, `ic50`, and `t_obs` now participate in forward
   or loss computation.
4. The primary cross-layer route remains detached and follows
   `tau_TI -> v0`; alpha coupling remains bounded and disabled by default.
5. A contract test checks finite losses, output shape, and graph signal use.

## Risks remaining for evidence review

- Synthetic checks establish software invariants only; they do not establish
  biological validity or task-level performance.
- The current graph implementation assumes one graph per batch. A production
  graph minibatcher must be added when samples have variable node counts.
- The model's mechanistic equations are regularizers and hypotheses until each
  equation is linked to a valid source and its domain assumptions are tested.
- Calibration, uncertainty intervals, external validation, and ablations must
  be completed with real data.

## Validation commands

```powershell
python train.py --smoke
python train.py --epochs 8 --stage-epochs 1
python verification\verify_math.py
python verification\verify_l4.py
python verification\contract.py
```
