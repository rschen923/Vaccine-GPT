# Upstream model weights

No pretrained or VaccineGPT-trained checkpoint is included in this repository.
Do not commit downloaded upstream weights here.

## v1

- ESM-2 default: `esm2_t33_650M_UR50D`, loaded with the official `fair-esm`
  package. The cache location is managed by PyTorch Hub; hash the actual
  checkpoint and record its source revision in `configs/model_manifest.json`
  and the embedding cache manifest before a training/validation run.
- CLEF reference/checkpoint: the code repository is
  `PengYueECUST1997/CLEF`; if using a released/local checkpoint, independently
  verify the expected architecture, license, source commit, and SHA-256. No
  verified CLEF checkpoint is currently staged.
- ESM-3 `esm3-sm-open-v1` is an optional separate runtime. It is not a
  replacement for an H1-H4 classifier and is not currently staged.

## v2

Evo 2 `evo2_7b` is a Linux/WSL2 CUDA-only optional backbone for DNA windows.
Its upstream runtime and checkpoint are not installed here. The Windows CPU
smoke test does not invoke Evo 2.

Production usage is blocked until the exact source revision, checkpoint hash,
license/access terms, inference device, and configuration are recorded.
