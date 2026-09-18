# VaccineGPT architecture dependencies

The new implementation in `VaccineGPT/vaccinegpt` intentionally uses only:

- Python 3.10 or newer
- PyTorch 2.1 or newer
- NumPy 1.23 or newer

The JSONL contract, grouped split checks, model, losses, PCGrad, curriculum,
and verification commands do not require pandas, scikit-learn, PyTorch
Geometric, transformers, or external foundation-model checkpoints. Optional
legacy scripts retain their existing dependency list in
`VaccineGPT/requirements.txt`; they are not imported by the new package.

From the repository root:

```powershell
python -m pip install torch numpy
python VaccineGPT/train.py synthetic --output VaccineGPT/data/synthetic.jsonl
python VaccineGPT/train.py validate VaccineGPT/data/synthetic.jsonl
python VaccineGPT/train.py train VaccineGPT/data/synthetic.jsonl --epochs 6
python -m verification.smoke
python -m verification.math
python -m verification.l4
```

