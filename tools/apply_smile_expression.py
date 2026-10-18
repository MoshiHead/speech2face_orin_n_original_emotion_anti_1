"""Calibrate, preview, and apply natural smiling expression to the avatar.

Usage:
    python tools/apply_smile_expression.py \
        --renderer_ckpt weights/renderer_narrow.pt \
        --alpha 1.0 \
        --preset natural \
        --preview_path assets/expression_preview.jpg \
        --delta_output assets/happy_expression_delta.pt

This tool applies the calibrated post-render smile expression deformation and
renders a genuine side-by-side preview of the avatar in neutral vs smiling.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NILOYS = ROOT / "src" / "niloys"
if str(NILOYS) not in sys.path:
    sys.path.insert(0, str(NILOYS))
LIVE = NILOYS / "live"
if str(LIVE) not in sys.path:
    sys.path.insert(0, str(LIVE))

from expression import PRESETS, get_smile_grid  # noqa: E402
from student.renderer import build as build_renderer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate and preview natural smile expression for the avatar"
    )
    parser.add_argument(
        "--renderer_ckpt",
        default="weights/renderer_narrow.pt",
        help="Path to distilled renderer checkpoint (renderer_narrow.pt)",
    )
    parser.add_argument(
        "--preview_path",
        default="assets/expression_preview.jpg",
        help="Path to save side-by-side neutral vs smiling preview image",
    )
    parser.add_argument(
        "--delta_output",
        default="assets/happy_expression_delta.pt",
        help="Path to save expression configuration / delta metadata",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Smile intensity multiplier (0=neutral, 0.65=subtle, 1.0=natural, 1.25=happy, 1.5=broad)",
    )
    parser.add_argument(
        "--preset",
        default="natural",
        choices=["subtle", "natural", "happy", "broad"],
        help="Smile preset profile",
    )
    args = parser.parse_args()

    print(f"Loading distilled renderer from: {args.renderer_ckpt}")
    if not os.path.exists(args.renderer_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.renderer_ckpt}")

    pack = torch.load(args.renderer_ckpt, map_location="cpu")
    renderer = build_renderer(**pack["cfg"]).eval()
    renderer.load_state_dict(pack["model"])

    print(f"Calibrating smile expression: alpha={args.alpha}, preset={args.preset}")
    grid = get_smile_grid(512, 512, alpha=args.alpha, preset=args.preset, device="cpu")

    # 1. Render side-by-side neutral vs smiling preview directly
    if args.preview_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.preview_path)), exist_ok=True)
        with torch.no_grad():
            m_neut = renderer.motion_mean.unsqueeze(0)
            img_neut = renderer(m_neut)  # [1, 3, 512, 512]

            if grid is not None:
                img_smile = F.grid_sample(
                    img_neut,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )[0]
            else:
                img_smile = img_neut[0]
            img_neut = img_neut[0]

        neut_np = (img_neut.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        smile_np = (img_smile.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())

        h, w, c = neut_np.shape
        side_by_side = np.zeros((h, w * 2, c), dtype=np.uint8)
        side_by_side[:, :w] = neut_np
        side_by_side[:, w:] = smile_np

        Image.fromarray(side_by_side).save(args.preview_path, quality=95)
        print(f"Side-by-side preview saved to: {args.preview_path}")

    # 2. Save expression metadata bundle
    if args.delta_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.delta_output)), exist_ok=True)
        payload = {
            "alpha": args.alpha,
            "preset": args.preset,
            "preset_scale": PRESETS.get(args.preset, 1.0),
            "effective_scale": args.alpha * PRESETS.get(args.preset, 1.0),
            "expression_delta": torch.zeros(32),
            "expression_mask": torch.ones(32),
        }
        torch.save(payload, args.delta_output)
        print(f"Expression metadata saved to: {args.delta_output}")

    print("\nSmile expression ready!")
    print(f"Run the live server with: EXPR_ALPHA={args.alpha} ./run_x86.sh")


if __name__ == "__main__":
    main()
