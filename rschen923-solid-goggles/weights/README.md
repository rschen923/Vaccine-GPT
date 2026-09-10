# Pretrained weight staging

This directory intentionally does not contain downloaded upstream checkpoints.
The model adapters load the original weights from the configured Hugging Face
identifier, installed upstream package, or local path.

Expected local artifact:

```text
weights/
  clef/
    checkpoint.pt
```

The CLEF checkpoint must match the imported `clef_enc` implementation and is
loaded with `strict=False` so a published training wrapper can be unwrapped
without changing the adapter. Record the exact source revision and SHA256 in
`configs/model_manifest.json` before a production run.
