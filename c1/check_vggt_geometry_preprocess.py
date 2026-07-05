#!/usr/bin/env python3
"""Simple VGGT-Omega geometry preprocessing check.

Usage:
  python c1/check_vggt_geometry_preprocess.py --image /path/to/camera.jpg

Correct result means:
  - dtype is torch.float32
  - min/max stay inside [0, 1]
  - height and width are multiples of 16
  - DrivoR online path and cache script both use VGGT-Omega official preprocessing
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DRIVOR_FEATURES = ROOT / "navsim/agents/drivoR/drivor_features.py"
CACHE_SCRIPT = ROOT / "navsim/agents/drivoR/scripts/cache_vggt_geometry_tokens.py"
AGGREGATOR = ROOT / "vggt_omega/models/aggregator.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check VGGT-Omega geometry image preprocessing.")
    parser.add_argument(
        "--image",
        default="/high_perf_store3/world-model/weixiaobao/yzj/DrivoR/dataset/sensor_blobs/trainval/2021.05.12.19.36.12_veh-35_00005_00204/CAM_B0/0fab98dd6dfb5032.jpg",
        type=Path,
        help="One raw camera image path.",
    )
    parser.add_argument("--mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    return parser.parse_args()


def check_source(path: Path, required: str, forbidden=()) -> None:
    text = path.read_text()
    missing = required not in text
    hits = [item for item in forbidden if item in text]
    status = "OK" if not missing and not hits else "BAD"
    print(f"{path.relative_to(ROOT)} source check: {status}")
    if missing:
        raise AssertionError(f"Missing required code text: {required}")
    if hits:
        raise AssertionError(f"Found forbidden old preprocessing code: {hits}")


def check_aggregator_normalize() -> None:
    text = AGGREGATOR.read_text()
    expected = "images = (images - self._resnet_mean) / self._resnet_std"
    has_buffers = "_RESNET_MEAN = [0.485, 0.456, 0.406]" in text and "_RESNET_STD = [0.229, 0.224, 0.225]" in text
    print(f"{AGGREGATOR.relative_to(ROOT)} internal normalize: {'OK' if expected in text and has_buffers else 'BAD'}")
    if expected not in text or not has_buffers:
        raise AssertionError("VGGT Aggregator should own the ImageNet mean/std normalization step.")


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    from navsim.agents.drivoR.vggt_geometry import preprocess_arrays_for_teacher

    raw = np.asarray(Image.open(args.image).convert("RGB"))
    images = preprocess_arrays_for_teacher(
        [raw],
        mode=args.mode,
        image_resolution=args.image_resolution,
        patch_size=args.patch_size,
    )
    one = images[0]
    print(
        f"vggt_geometry_preprocess: shape={tuple(images.shape)} dtype={one.dtype} "
        f"min={one.min().item():.6f} max={one.max().item():.6f} mean={one.mean().item():.6f}"
    )

    if images.ndim != 4 or images.shape[1] != 3:
        raise AssertionError(f"Expected shape [N, 3, H, W], got {tuple(images.shape)}")
    if one.dtype != torch.float32:
        raise AssertionError(f"Expected torch.float32, got {one.dtype}")
    if one.min().item() < -1e-6 or one.max().item() > 1.0 + 1e-6:
        raise AssertionError("VGGT geometry input should be in [0, 1]. Do not pre-normalize before VGGT.")
    if images.shape[-2] % args.patch_size != 0 or images.shape[-1] % args.patch_size != 0:
        raise AssertionError("VGGT geometry input height/width should be multiples of patch size 16.")

    check_source(DRIVOR_FEATURES, "preprocess_arrays_for_teacher", forbidden=("_preprocess_vggt_geometry_image",))
    check_source(CACHE_SCRIPT, "load_and_preprocess_images", forbidden=("resize_center_crop", "resize_letterbox"))
    check_aggregator_normalize()

    print("PASS: VGGT geometry feeds [0,1] RGB images; VGGT Aggregator performs the single mean/std normalization.")


if __name__ == "__main__":
    main()
