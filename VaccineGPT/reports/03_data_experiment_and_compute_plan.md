# Public data, experiments, and Tencent Cloud GPU plan

## Data plan

Use versioned, checksum-backed public sources already represented by the
repository manifests, then map each source to the authoritative tensor
contract. Candidate source families include RefSeq/UniProt for sequence,
STRING/other PPI resources for graph edges, VFDB/OGEE/DEG for virulence and
essentiality labels, IEDB for epitope/immunogenicity evidence, and SRA or
single-cell repositories for expression and immune-state validation. License,
release, identifier mapping, negative controls, homology overlap, batch, and
missing-label policy must be recorded per source.

## Computational experiments

1. Grouped strain/species split with homology-aware leakage audit.
2. External-source evaluation held out by source and organism.
3. Ablations for graph signal, each detached coupling, each physical
   consistency term, foundation representation, and PCGrad.
4. Calibration and uncertainty evaluation using held-out confidence labels.
5. Seed and hyperparameter sensitivity with pre-registered acceptance metrics.

## Experimental validation

Before wet-lab execution, obtain institutional biosafety and ethics approval.
Use non-pathogenic or approved strains, randomized treatment assignment,
negative/vehicle controls, positive controls, blinded readout where feasible,
biological replicates, and pre-specified endpoints. Validate predictions with
orthogonal assays rather than treating model scores as mechanistic proof.
Exact sample sizes and assay parameters must be populated from included
recent full-text studies in the evidence table.

## Compute estimate (planning estimate, not a Tencent quotation)

Start with one 40 GB-class GPU for smoke, pilot, ablations, and one-seed
training. Reserve multi-GPU only for the L0 representation pretraining stage
after profiling. Record GPU model, hours, memory peak, storage, data transfer,
checkpoint count, and retry overhead. The executable training commands are:

```powershell
python train.py --data path\to\contract_batch.jsonl --epochs 8 --stage-epochs 1 --checkpoint checkpoints\five_layer.pt
```

The final budget must be recalculated from measured throughput and the chosen
Tencent Cloud instance price at procurement time; this repository does not
invent a current price.
