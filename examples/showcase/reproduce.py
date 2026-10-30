#!/usr/bin/env python3
"""Reproduce one or all public SIMPLEPOSTER showcase cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("cases.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="case id, or 'all'")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "showcase")
    parser.add_argument("--verify-only", action="store_true", help="verify assets and print commands without inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    settings = manifest["settings"]
    cases = manifest["cases"]
    if args.case != "all":
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            raise SystemExit(f"unknown case: {args.case}")

    if args.checkpoint.exists():
        actual = sha256(args.checkpoint)
        expected = manifest["model_sha256"]
        if actual != expected:
            raise SystemExit(f"checkpoint SHA256 mismatch: expected {expected}, got {actual}")
    elif not args.verify_only:
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="simpleposter-showcase-") as temp_dir:
        temp = Path(temp_dir)
        for case in cases:
            case_settings = {**settings, **case.get("settings", {})}
            for key in ("input", "mask", "reference_output"):
                path = ROOT / case[key]
                actual = sha256(path)
                expected = case["sha256"][key]
                if actual != expected:
                    raise SystemExit(f"{case['id']} {key} SHA256 mismatch")

            ocr_path = temp / f"{case['id']}.json"
            ocr_path.write_text(
                json.dumps(case["ocr_result"], ensure_ascii=False), encoding="utf-8"
            )
            command = [
                sys.executable,
                str(ROOT / "infer.py"),
                "--image", str(ROOT / case["input"]),
                "--mask", str(ROOT / case["mask"]),
                "--ocr-json", str(ocr_path),
                "--prompt", case["prompt"],
                "--checkpoint", str(args.checkpoint),
                "--vae", str(args.vae),
                "--qwen-dir", str(args.qwen_dir),
                "--output", str(args.output_dir / f"{case['id']}.png"),
                "--width", str(case_settings["width"]),
                "--height", str(case_settings["height"]),
                "--steps", str(case_settings["steps"]),
                "--guidance-scale", str(case_settings["guidance_scale"]),
                "--guidance", str(case_settings["guidance"]),
                "--seed", str(case_settings["seed"]),
            ]
            print(" ".join(json.dumps(part, ensure_ascii=False) for part in command))
            if not args.verify_only:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
