"""Extract the expression delta between happy and neutral demo videos.

Usage:
    python tools/extract_expression_delta.py \
        --happy_video emotion_demo_videos/happy.mp4 \
        --neutral_video emotion_demo_videos/neutral.mp4 \
        --renderer_path weights/renderer.ckpt \
        --student_ckpt weights/motion_student_38M.pt \
        --output assets/happy_expression_delta.pt

The full IMTRenderer's MotionEncoder extracts 32-D motion latents from each
frame.  The per-dimension MEAN shift between happy and neutral videos gives
the expression direction.  A soft mask built from variance analysis protects
dimensions that are phoneme-dependent (high cross-frame variance in the delta)
from being overwritten.

The delta is saved in the CausalMotionStudent's NORMALIZED motion space so it
can be added directly to the student's output before denorm_motion().
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ---- paths ----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMTALKER = ROOT / "src" / "imtalker"
NILOYS = ROOT / "src" / "niloys"
for p in (str(IMTALKER), str(NILOYS)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_renderer(renderer_path: str, device: torch.device):
    """Load the full IMTRenderer (we only need its MotionEncoder)."""
    from types import SimpleNamespace
    args = SimpleNamespace(
        input_size=512,
        swin_res_threshold=128,
        num_heads=8,
        window_size=8,
    )
    from renderer.models import IMTRenderer
    renderer = IMTRenderer(args).to(device).eval()
    ckpt = torch.load(renderer_path, map_location="cpu")
    raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {
        k.replace("gen.", "", 1).replace("model.", "", 1): v
        for k, v in raw.items()
    }
    renderer.load_state_dict(cleaned, strict=False)
    for p in renderer.parameters():
        p.requires_grad_(False)
    return renderer


def _extract_motions_from_video(
    video_path: str,
    motion_encoder: torch.nn.Module,
    device: torch.device,
    max_frames: int = 0,
) -> torch.Tensor:
    """Extract 32-D motion latent from every frame of a video.

    Returns [N, 32] tensor of raw motion latents.
    """
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    motions = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            mot = motion_encoder(tensor)  # [1, 32]
        motions.append(mot.cpu())
        count += 1
        if max_frames > 0 and count >= max_frames:
            break
    cap.release()
    if not motions:
        raise RuntimeError(f"No frames extracted from {video_path}")
    result = torch.cat(motions, dim=0)  # [N, 32]
    print(f"    {video_path}: {result.shape[0]} frames, "
          f"motion range [{result.min():.4f}, {result.max():.4f}]")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract expression delta between happy and neutral videos"
    )
    parser.add_argument("--happy_video", required=True,
                        help="Path to the happy expression video")
    parser.add_argument("--neutral_video", required=True,
                        help="Path to the neutral expression video")
    parser.add_argument("--renderer_path", required=True,
                        help="Path to the full IMTRenderer checkpoint (renderer.ckpt)")
    parser.add_argument("--student_ckpt", default="",
                        help="Path to the CausalMotionStudent checkpoint "
                             "(for normalization stats; optional)")
    parser.add_argument("--output", required=True,
                        help="Output .pt file path")
    parser.add_argument("--mask_percentile", type=float, default=60.0,
                        help="Percentile of delta-variance below which a dimension "
                             "is considered expression-only (protected from zeroing). "
                             "Higher = more dimensions contribute to expression")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames per video (0 = all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load the full renderer's MotionEncoder
    print("Loading full IMTRenderer (MotionEncoder) ...")
    renderer = _load_renderer(args.renderer_path, device)
    motion_encoder = renderer.latent_token_encoder
    print(f"    MotionEncoder output dim: "
          f"{renderer.latent_token_encoder.final_linear.out_features}")

    # 2. Extract motions from both videos
    print("Extracting motions from happy video ...")
    happy_motions = _extract_motions_from_video(
        args.happy_video, motion_encoder, device, args.max_frames
    )  # [Nh, 32]

    print("Extracting motions from neutral video ...")
    neutral_motions = _extract_motions_from_video(
        args.neutral_video, motion_encoder, device, args.max_frames
    )  # [Nn, 32]

    # 3. Compute expression delta and mask
    happy_mean = happy_motions.mean(dim=0)      # [32]
    neutral_mean = neutral_motions.mean(dim=0)  # [32]
    delta_raw = happy_mean - neutral_mean       # [32] in raw motion space

    print(f"\nRaw expression delta:")
    print(f"    norm = {delta_raw.norm():.4f}")
    print(f"    per-dim abs range: [{delta_raw.abs().min():.4f}, "
          f"{delta_raw.abs().max():.4f}]")
    print(f"    top-5 dims by |delta|: "
          f"{delta_raw.abs().topk(5).indices.tolist()}")

    # Variance-based mask: for each dimension, compute the variance across
    # frames.  High variance means the dimension is phoneme-dependent and
    # should NOT be shifted.
    happy_var = happy_motions.var(dim=0)      # [32]
    neutral_var = neutral_motions.var(dim=0)  # [32]
    combined_var = (happy_var + neutral_var) / 2.0  # [32]

    # Soft mask: dimensions with LOW variance get mask ~ 1 (safe to shift),
    # dimensions with HIGH variance get mask ~ 0 (phoneme-dependent, protect).
    var_threshold = torch.quantile(
        combined_var, args.mask_percentile / 100.0
    ).item()
    sharpness = 5.0 / max(var_threshold, 1e-6)
    expression_mask = torch.sigmoid(
        sharpness * (var_threshold - combined_var)
    )  # [32], values in (0, 1)

    print(f"\nExpression mask:")
    print(f"    variance threshold (p{args.mask_percentile:.0f}): {var_threshold:.6f}")
    print(f"    mask range: [{expression_mask.min():.4f}, {expression_mask.max():.4f}]")
    print(f"    dims with mask > 0.5: "
          f"{(expression_mask > 0.5).sum().item()}/32")
    print(f"    dims with mask > 0.8: "
          f"{(expression_mask > 0.8).sum().item()}/32")

    # 4. Convert delta to the student's normalized motion space
    # Student works in normalized space: norm = (raw - mean) / std
    # A shift of delta_raw in raw space = delta_raw / std in normalized space
    expression_delta_normalized = delta_raw.clone()
    if args.student_ckpt:
        print(f"\nLoading student normalization stats from {args.student_ckpt} ...")
        pack = torch.load(args.student_ckpt, map_location="cpu")
        if "model" in pack:
            sd = pack["model"]
            motion_mean = sd.get("motion_mean", torch.zeros(32))
            motion_std = sd.get("motion_std", torch.ones(32))
        else:
            motion_mean = torch.zeros(32)
            motion_std = torch.ones(32)
        motion_std = motion_std.clamp_min(1e-5)
        expression_delta_normalized = delta_raw / motion_std
        print(f"    motion_mean range: [{motion_mean.min():.4f}, "
              f"{motion_mean.max():.4f}]")
        print(f"    motion_std range: [{motion_std.min():.4f}, "
              f"{motion_std.max():.4f}]")
        print(f"    normalized delta norm: "
              f"{expression_delta_normalized.norm():.4f}")
    else:
        print("\nNo student checkpoint provided; delta saved in raw space.")
        print("    Pass --student_ckpt to convert to normalized space.")

    # 5. Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {
        "expression_delta": expression_delta_normalized,  # [32]
        "expression_mask": expression_mask,                # [32]
        "delta_raw": delta_raw,                            # [32] raw space
        "happy_mean": happy_mean,                          # [32]
        "neutral_mean": neutral_mean,                      # [32]
        "combined_var": combined_var,                       # [32]
        "mask_percentile": args.mask_percentile,
        "happy_frames": happy_motions.shape[0],
        "neutral_frames": neutral_motions.shape[0],
        "happy_video": str(args.happy_video),
        "neutral_video": str(args.neutral_video),
        "raw_happy_motions": happy_motions,                # [Nh, 32]
        "raw_neutral_motions": neutral_motions,            # [Nn, 32]
    }
    torch.save(payload, args.output)
    print(f"\nSaved expression delta to {args.output}")
    print(f"    expression_delta: {tuple(expression_delta_normalized.shape)}")
    print(f"    expression_mask: {tuple(expression_mask.shape)}")
    print("\nDone.  Use --expression_delta_path in "
          "imtalker_try19_student_live.py to apply.")


if __name__ == "__main__":
    main()
