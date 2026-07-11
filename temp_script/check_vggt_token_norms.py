"""Report the per-token L2-norm distribution of a cached VGGT-Omega geometry token set.

Why: high-norm outlier registers act as attention sinks in decoder cross-attention
(softmax mass concentrates on large-norm keys). This script measures how uneven the
cached token norms are -- overall, per frame/global half, and per (camera, register)
position -- to document whether the pre-projection LayerNorm in VggtGeometryProjector
is load-bearing in geo_only memory mode (see code_change_md/design/geo_only.md,
acceptance item 4).

Usage:
    python check_vggt_token_norms.py --cache-dir <token cache dir> [--max-samples 2000]
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from navsim.agents.drivoR.vggt_geometry import (
    VGGT_GEOMETRY_SCENE_DICT_KEYS,
    _load_token_index,
    _select_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect per-token norms of a VGGT geometry token cache.")
    parser.add_argument("--cache-dir", required=True, help="Token cache directory (contains token_index.json / *.pt).")
    parser.add_argument("--max-samples", type=int, default=2000, help="Number of samples to inspect (0 = all).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    return parser.parse_args()


def resolve_sample_paths(cache_dir: Path) -> dict:
    index = _load_token_index(cache_dir)
    if index is not None:
        return {token: cache_dir / rel for token, rel in sorted(index.items())}
    return {
        path.stem: path
        for path in sorted(cache_dir.rglob("*.pt"))
        if ".tmp." not in path.name and path.name != "noise_stats.pt"
    }


def summarize(name: str, norms: torch.Tensor) -> None:
    flat = norms.reshape(-1)
    quantiles = torch.tensor([0.01, 0.50, 0.90, 0.99, 0.999])
    p1, p50, p90, p99, p999 = torch.quantile(flat, quantiles).tolist()
    print(f"\n[{name}] tokens={flat.numel()}")
    print(f"  mean={flat.mean():.2f}  p1={p1:.2f}  p50={p50:.2f}  p90={p90:.2f}  p99={p99:.2f}  p99.9={p999:.2f}  max={flat.max():.2f}")
    print(f"  ratios: p99/p50={p99 / p50:.2f}  max/p50={flat.max().item() / p50:.2f}")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser()
    paths = resolve_sample_paths(cache_dir)
    if not paths:
        raise FileNotFoundError(f"No cached token .pt files found in {cache_dir}")

    tokens = sorted(paths.keys())
    if args.max_samples and len(tokens) > args.max_samples:
        rng = torch.Generator().manual_seed(args.seed)
        picks = torch.randperm(len(tokens), generator=rng)[: args.max_samples].tolist()
        tokens = [tokens[i] for i in sorted(picks)]

    full_norms, frame_norms, global_norms = [], [], []
    for token in tqdm(tokens, desc="Computing token norms"):
        tensor = _select_tensor(torch.load(paths[token], map_location="cpu")).float()  # (4, T, D)
        half = tensor.shape[-1] // 2
        full_norms.append(tensor.norm(dim=-1))
        frame_norms.append(tensor[..., :half].norm(dim=-1))
        global_norms.append(tensor[..., half:].norm(dim=-1))

    full = torch.stack(full_norms)      # (S, 4, T)
    frame = torch.stack(frame_norms)
    glob = torch.stack(global_norms)
    num_samples, num_cams, tokens_per_cam = full.shape

    print(f"\nCache: {cache_dir}")
    print(f"Samples inspected: {num_samples}   shape per sample: ({num_cams}, {tokens_per_cam}, D)")
    if tokens_per_cam == 17:
        print("Note: 17 tokens per camera => position 0 is the camera token, 1-16 are registers.")

    summarize("full 2048-dim", full)
    summarize("frame half (dims 0:1024)", frame)
    summarize("global half (dims 1024:2048)", glob)

    print("\nPer-position mean norm (rows = cameras, cols = register positions):")
    position_mean = full.mean(dim=0)  # (4, T)
    overall_median = full.median()
    header = "        " + " ".join(f"{i:>7d}" for i in range(tokens_per_cam))
    print(header)
    for cam_index in range(num_cams):
        label = VGGT_GEOMETRY_SCENE_DICT_KEYS[cam_index] if cam_index < len(VGGT_GEOMETRY_SCENE_DICT_KEYS) else f"CAM_{cam_index}"
        row = " ".join(f"{value:7.1f}" for value in position_mean[cam_index].tolist())
        print(f"{label:>7s} {row}")

    hot = (position_mean > 2 * overall_median).nonzero(as_tuple=False)
    if len(hot):
        positions = ", ".join(f"(cam={c.item()}, pos={p.item()}, mean={position_mean[c, p]:.1f})" for c, p in hot)
        print(f"\nPositions with mean norm > 2x overall median: {positions}")
    else:
        print("\nNo position has mean norm > 2x the overall median.")

    print("\nTop-10 outlier tokens (sample, camera, position, norm):")
    top_values, top_indices = full.reshape(-1).topk(min(10, full.numel()))
    for value, flat_index in zip(top_values.tolist(), top_indices.tolist()):
        sample_index, remainder = divmod(flat_index, num_cams * tokens_per_cam)
        cam_index, position = divmod(remainder, tokens_per_cam)
        print(f"  {tokens[sample_index]}  cam={cam_index}  pos={position}  norm={value:.1f}")

    print("\nReading guide: p99/p50 well above ~3, or hot positions above, indicate attention-sink-prone")
    print("outlier registers -- the pre-projection LayerNorm is doing real work and should stay.")


if __name__ == "__main__":
    main()
