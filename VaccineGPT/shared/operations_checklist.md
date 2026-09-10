# Shared operations checklist for M1 / M2

## 1. Data lineage

- Assign and record `model_version`, `feature_version`, `label_version`, `batch_id` for every artifact.
- Store lineage metadata next to each feature cache, label export, checkpoint, and evaluation report.
- Validate compatibility before model loading and before training-run restart.

## 2. M1 validation gates

- Confirm all view inputs share the expected shape contract.
- Ensure fused representation dimension equals 128.
- Validate that the RGCN-like message pass stays stable across batches.
- Test SOM and INT branch behavior independently.

## 3. M2 validation gates

- Confirm T1 is three-class, T2/T3 are scalar prediction outputs, and T4 ranking is evaluated per candidate group.
- Validate multi-task loss returns finite values during training.
- Check NDCG@10 and MRR against the same sample group boundaries.

## 4. Runtime fallback

If any external pretrained encoder is unavailable (ESM-2, DNABERT-2, RNA-FM, ProteomeLM), switch to cached feature mode. Keep the same `UDC` schema and add a comment in the calling script explaining:

- when the cache mode is triggered,
- which model family is skipped,
- how to resume in full mode later.

## 5. Handoff assets for M3-M6

- `h_final` (`float32`, 128 dimensions) is the M1 handoff to M2/M3/M4.
- M2 emits T1/T2/T3 prediction outputs and T4 candidate scores.
- Every artifact must carry the lineage tuple and a reproducible input split.
- Future modules should consume UDC records rather than importing M1 internals.
- Wet-lab feedback should map to `gene_id`, task, label level, source, and lineage.

## 6. Artifact outputs

- `features/`
- `labels/`
- `checkpoints/`
- `reports/`
- `lineage/`
