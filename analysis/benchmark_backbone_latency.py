#!/usr/bin/env python3
"""Benchmark current DrivoR image backbone latency.

This measures ImgEncoder only. It does not include C1 online VGGT teacher cost.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from drivor_analysis_utils import build_image_backbone_config, merge_lora_to_backbone
from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import ImgEncoder


class PruningBlock(nn.Module):
    """Prune patch tokens after attention so MLP only sees kept prefix tokens."""

    def __init__(self, base_block: nn.Module, keep_tokens: int):
        super().__init__()
        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self.ls1 = getattr(base_block, "ls1", nn.Identity())
        self.drop_path1 = getattr(base_block, "drop_path1", nn.Identity())
        self.norm2 = base_block.norm2
        self.mlp = base_block.mlp
        self.ls2 = getattr(base_block, "ls2", nn.Identity())
        self.drop_path2 = getattr(base_block, "drop_path2", nn.Identity())
        self.keep_tokens = keep_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x[:, : self.keep_tokens, :]
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DrivoR image backbone latency")
    parser.add_argument(
        "--agent-config",
        type=str,
        default="navsim/planning/script/config/common/agent/drivoR.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cams", type=int, default=4)
    parser.add_argument("--scene-tokens", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--merge-lora", action="store_true", default=True)
    parser.add_argument("--no-merge-lora", action="store_false", dest="merge_lora")
    parser.add_argument("--prune-last-blocks", type=int, default=0)
    return parser.parse_args()


def build_backbone(agent_config: str, num_scene_tokens: int, device: torch.device, merge_lora: bool) -> ImgEncoder:
    cfg = build_image_backbone_config(agent_config, num_scene_tokens=num_scene_tokens)
    backbone = ImgEncoder(cfg).to(device)
    backbone.analysis_image_size = tuple(cfg.image_size)
    backbone.eval()
    backbone.use_grid_mask = False
    if merge_lora:
        merge_lora_to_backbone(backbone)
    return backbone


def apply_block_pruning(backbone: nn.Module, keep_tokens: int, num_last_blocks: int) -> None:
    if num_last_blocks <= 0:
        return
    vit = backbone.model
    blocks = list(vit.blocks)
    start = max(0, len(blocks) - num_last_blocks)
    for idx in range(start, len(blocks)):
        blocks[idx] = PruningBlock(blocks[idx], keep_tokens=keep_tokens)
    vit.blocks = nn.Sequential(*blocks)


def benchmark_backbone(
    backbone: nn.Module,
    device: torch.device,
    batch_size: int,
    num_cams: int,
    num_scene_tokens: int,
    image_size: Tuple[int, int],
    num_iters: int,
    warmup: int,
) -> float:
    width, height = image_size
    embed_dim = backbone.num_features
    image = torch.randn(batch_size, num_cams, 3, height, width, device=device)
    scene_tokens = torch.randn(batch_size, num_cams, num_scene_tokens, embed_dim, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = backbone(image, scene_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(num_iters):
                _ = backbone(image, scene_tokens)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_iters

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = backbone(image, scene_tokens)
    end = time.perf_counter()
    return (end - start) / num_iters * 1000.0


def run_sweep(args, device: torch.device, pruned: bool = False) -> List[Tuple[int, float, float]]:
    rows = []
    for n_tokens in args.scene_tokens:
        backbone = build_backbone(args.agent_config, n_tokens, device, args.merge_lora)
        if pruned:
            apply_block_pruning(backbone, keep_tokens=n_tokens, num_last_blocks=args.prune_last_blocks)
        image_size = tuple(backbone.analysis_image_size)
        latency_ms = benchmark_backbone(
            backbone,
            device,
            args.batch_size,
            args.num_cams,
            n_tokens,
            image_size,
            args.iters,
            args.warmup,
        )
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        rows.append((n_tokens, latency_ms, fps))
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def print_table(title: str, rows: List[Tuple[int, float, float]], baseline=None) -> None:
    print(f"\n{title}")
    if baseline is None:
        print(f"{'scene_tokens':>14} | {'latency_ms':>12} | {'FPS':>8}")
        print("-" * 42)
        for n_tokens, latency, fps in rows:
            print(f"{n_tokens:>14} | {latency:>12.3f} | {fps:>8.1f}")
    else:
        baseline_map = {n: lat for n, lat, _fps in baseline}
        print(f"{'scene_tokens':>14} | {'latency_ms':>12} | {'FPS':>8} | {'speedup':>8}")
        print("-" * 55)
        for n_tokens, latency, fps in rows:
            speedup = baseline_map[n_tokens] / latency if latency > 0 and n_tokens in baseline_map else 0.0
            print(f"{n_tokens:>14} | {latency:>12.3f} | {fps:>8.1f} | {speedup:>7.2f}x")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[benchmark] device={device}")
    print("[benchmark] scope=image backbone only; C1 online VGGT teacher cost is not included")

    original_rows = run_sweep(args, device, pruned=False)
    print_table("Original", original_rows)

    if args.prune_last_blocks > 0:
        pruned_rows = run_sweep(args, device, pruned=True)
        print_table(f"Pruned last {args.prune_last_blocks} block(s)", pruned_rows, baseline=original_rows)


if __name__ == "__main__":
    main()
