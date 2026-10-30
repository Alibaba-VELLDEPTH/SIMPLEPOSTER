#!/usr/bin/env python3
"""Local-only inference entry point for the simple poster model."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from aigc_train.models.flux.autoencoder import AutoEncoder, AutoEncoderParams
from aigc_train.models.flux.flux import FluxParams, FluxWithConnector
from aigc_train.models.flux.layers import (
    DoubleStreamBlockLoraProcessor,
    SingleStreamBlockLoraProcessor,
)
from aigc_train.models.flux.xflux_pipe import XFluxMLLMInpPipeline
from aigc_train.models.qwenvl.conditioner import Qwen25VL_7b_Embedder_textIds
from aigc_train.models.qwenvl.connector_edit import Qwen2Connector


LOGGER = logging.getLogger("simple_poster")
DEFAULT_NEGATIVE_PROMPT = (
    "watermark, signature, text font, blurry, bad quality, blurry, bad, ugly, "
    "terrible, painting, 3d, render, comic, anime, bad hands, bad anatomy, "
    "bad body, bad face, bad teeth, bad arms, bad legs, deformities"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple-poster inference locally on one GPU.")
    parser.add_argument("--image", type=Path, required=True, help="RGBA product image, or RGB when --mask is set")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local dit.pth")
    parser.add_argument("--vae", type=Path, required=True, help="Local ae.safetensors")
    parser.add_argument("--qwen-dir", type=Path, required=True, help="Local Qwen2.5-VL-7B-Instruct directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/poster.png"))
    parser.add_argument("--mask", type=Path, help="Optional mask; white pixels are regenerated")
    parser.add_argument("--ocr-json", type=Path, help="Optional OCR result with texts, boxes and image size")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--guidance", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="Allow missing or unexpected checkpoint keys (off by default)",
    )
    args = parser.parse_args()

    if args.width % 16 or args.height % 16:
        parser.error("--width and --height must be divisible by 16")
    for name in ("image", "checkpoint", "vae", "qwen_dir"):
        path = getattr(args, name)
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.mask and not args.mask.exists():
        parser.error(f"--mask does not exist: {args.mask}")
    if args.ocr_json and not args.ocr_json.exists():
        parser.error(f"--ocr-json does not exist: {args.ocr_json}")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        parser.error("CUDA is required for this model")
    return args


def add_flux_lora_for_inference(dit: FluxWithConnector, rank: int = 256) -> FluxWithConnector:
    processors = {}
    for name, current in dit.attn_processors.items():
        match = re.search(r"\.(\d+)\.", name)
        layer_index = int(match.group(1)) if match else -1
        if name.startswith("double_blocks") and layer_index < 19:
            processors[name] = DoubleStreamBlockLoraProcessor(
                dim=3072,
                rank=rank,
                img_only=False,
                mlp_lora=True,
                mlp_hidden_size=3072 * 4,
            )
        elif name.startswith("single_blocks") and layer_index < 38:
            processors[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
        else:
            processors[name] = current
    dit.set_attn_processor(processors)
    dit.requires_grad_(False)
    dit.eval()
    return dit


def load_dit(checkpoint: Path, allow_mismatch: bool) -> FluxWithConnector:
    params = FluxParams(
        in_channels=384,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    )
    dit = FluxWithConnector(
        connector=Qwen2Connector(),
        params=params,
        enable_gradient_checkpointing=False,
    )
    dit = add_flux_lora_for_inference(dit)
    LOGGER.info("Loading DiT checkpoint: %s", checkpoint)
    state_dict = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if (missing or unexpected) and not allow_mismatch:
        raise RuntimeError(
            "Checkpoint does not match the verified model topology. "
            f"missing={missing}, unexpected={unexpected}"
        )
    if missing or unexpected:
        LOGGER.warning("Checkpoint mismatch allowed: missing=%s unexpected=%s", missing, unexpected)
    return dit.to(dtype=torch.bfloat16)


def load_vae(path: Path) -> AutoEncoder:
    params = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    vae = AutoEncoder(params)
    LOGGER.info("Loading VAE: %s", path)
    state_dict = load_file(str(path), device="cpu")
    vae.load_state_dict(state_dict, strict=True, assign=True)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def load_qwen(path: Path, device: str) -> Qwen25VL_7b_Embedder_textIds:
    LOGGER.info("Loading Qwen model: %s", path)
    processor = AutoProcessor.from_pretrained(
        str(path),
        min_pixels=256 * 28 * 28,
        max_pixels=324 * 28 * 28,
        local_files_only=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    return Qwen25VL_7b_Embedder_textIds(processor=processor, model=model, device=device)


def load_image_and_mask(image_path: Path, mask_path: Path | None, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    source = Image.open(image_path)
    if mask_path:
        mask = Image.open(mask_path).convert("L")
    elif source.mode == "RGBA":
        mask = ImageOps.invert(source.getchannel("A"))
    else:
        raise ValueError("RGB input requires --mask; otherwise provide an RGBA image with transparency")

    image = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    mask = mask.resize(size, Image.Resampling.NEAREST).point(lambda p: 255 if p > 128 else 0).convert("RGB")
    return image, mask


def load_ocr(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "ocr_result" in payload:
        return payload["ocr_result"]
    return payload


def build_text_positions(ocr: dict[str, Any], prompt: str, fallback_size: tuple[int, int]) -> dict[str, dict[int, list[float]]]:
    info = ocr.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
    width = float(info.get("width") or fallback_size[0])
    height = float(info.get("height") or fallback_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("OCR width and height must be positive")

    text_to_box = dict(zip(ocr.get("texts", []), ocr.get("boxes", [])))
    result: defaultdict[str, dict[int, list[float]]] = defaultdict(dict)
    for match in re.finditer(r"“([^”]+)”", prompt):
        phrase = match.group(1)
        box = text_to_box.get(phrase)
        if not box or len(box) < 8 or not phrase:
            continue
        left, top, right, bottom = float(box[0]), float(box[1]), float(box[2]), float(box[7])
        char_width = (right - left) / len(phrase)
        center_y = (top + bottom) / 2
        for index, char in enumerate(phrase):
            absolute_index = match.start(1) + index
            center_x = left + (index + 0.5) * char_width
            result[char][absolute_index] = [round(center_x / width, 4), round(center_y / height, 4)]
    return dict(result)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    size = (args.width, args.height)
    image, mask = load_image_and_mask(args.image, args.mask, size)
    text_positions = None
    if args.ocr_json:
        text_positions = build_text_positions(load_ocr(args.ocr_json), args.prompt, size)

    dit = load_dit(args.checkpoint, args.allow_checkpoint_mismatch)
    vae = load_vae(args.vae)
    qwen = load_qwen(args.qwen_dir, args.device)
    pipe = XFluxMLLMInpPipeline(dit=dit, mllm=qwen, ae=vae, device=args.device)

    LOGGER.info("Generating %sx%s poster with %s steps", args.width, args.height, args.steps)
    result = pipe(
        image=image,
        mask=mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_steps=args.steps,
        true_gs=args.guidance_scale,
        guidance=args.guidance,
        text_postions=text_positions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    LOGGER.info("Saved: %s", args.output.resolve())


if __name__ == "__main__":
    main()
