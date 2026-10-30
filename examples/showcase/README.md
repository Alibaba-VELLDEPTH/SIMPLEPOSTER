# Reproducible showcase

These six cases were selected from the evaluation set because they combine dense or mixed-format text with exact OCR matches. Every target line in every reference output matched exactly (`mean NED = 1.0`).

The directory includes the RGB product inputs, explicit generation masks, text-layout coordinates, full prompts, fixed sampling settings, and SHA256 checksums. `cases.json` is the machine-readable source of truth.

## Run one case

```bash
python examples/showcase/reproduce.py \
  --case net \
  --checkpoint models/dit.pth \
  --vae models/ae.safetensors \
  --qwen-dir models/Qwen2.5-VL-7B-Instruct
```

Use `--case all` to run all six. Use `--verify-only` to validate the checkpoint and all bundled assets without loading a model.

The verified settings are 1024×1024, 50 steps, guidance scale 4.5, guidance 30, and seed 100. The expected checkpoint SHA256 is recorded in `cases.json`. GPU kernels and dependency versions can introduce small pixel-level differences, so validation should use the target text and visual composition rather than requiring byte-identical PNG files.

White mask pixels are regenerated; black pixels preserve the product.
