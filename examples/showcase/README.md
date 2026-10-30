# Reproducible showcase

The showcase contains six benchmark cases with exact OCR matches (`mean NED = 1.0`) and one separately identified complex-tier case. The complex-tier example was checked target by target: all 10 requested text blocks are visible in the reference output.

The directory includes the RGB product inputs, explicit generation masks, text-layout coordinates, full prompts, fixed sampling settings, and SHA256 checksums. `cases.json` is the machine-readable source of truth.

## Run one case

```bash
python examples/showcase/reproduce.py \
  --case net \
  --checkpoint models/dit.pth \
  --vae models/ae.safetensors \
  --qwen-dir models/Qwen2.5-VL-7B-Instruct
```

Use `--case complex_laminator` for the complex-tier highlight, or `--case all` to run every bundled case. Use `--verify-only` to validate the checkpoint and all bundled assets without loading a model.

The benchmark cases use 1024×1024. The complex-tier case uses its verified 768×768 setting. All cases use 50 steps, guidance scale 4.5, guidance 30, and seed 100. The expected checkpoint SHA256 is recorded in `cases.json`. GPU kernels and dependency versions can introduce small pixel-level differences, so validation should use the target text and visual composition rather than requiring byte-identical PNG files.

White mask pixels are regenerated; black pixels preserve the product.
