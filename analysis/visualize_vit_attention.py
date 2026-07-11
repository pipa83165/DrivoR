#!/usr/bin/env python3
"""Visualize DrivoR ViT attention maps on current NAVSIM data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from drivor_analysis_utils import (
    build_dataset,
    build_scene_loader,
    disable_backbone_grid_mask,
    first_existing_tensor,
    get_drivor_camera_names,
    get_real_vit_from_backbone,
    instantiate_agent,
    load_training_config,
    make_dataloader,
    merge_lora_to_backbone,
    move_to_device,
    parse_int_list,
    set_cfg_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DrivoR ViT attention")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--config_path",
        type=str,
        default="navsim/planning/script/config/training/default_training.yaml",
    )
    parser.add_argument("--split", type=str, default="navtrain")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sensor_blobs_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./analysis_output/attention_viz")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=1)
    parser.add_argument("--layers", type=str, default=None, help="Comma list, e.g. 0,4,11. Default: all")
    parser.add_argument("--heads", type=str, default=None, help="Comma list. Default: first 4 heads")
    parser.add_argument("--queries", type=str, default="0", help="Scene-token query indices by default")
    parser.add_argument("--samples", type=str, default="0", help="Batch sample indices to render")
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hydra_override", action="append", default=[])
    return parser.parse_args()


def register_attention_hooks(vit_model, layer_indices: Optional[List[int]] = None):
    attn_storage: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs, _output):
            x = inputs[0]
            batch, tokens, _channels = x.shape
            qkv = module.qkv(x)
            num_heads = getattr(module, "num_heads", None) or getattr(module, "heads", 8)
            head_dim = getattr(module, "head_dim", None)
            if head_dim is None:
                head_dim = (qkv.shape[-1] // 3) // num_heads
            qkv = qkv.reshape(batch, tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            scale = getattr(module, "scale", head_dim**-0.5)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn_storage[layer_idx] = attn.softmax(dim=-1).detach().cpu()

        return hook

    for idx, blk in enumerate(vit_model.blocks):
        if layer_indices is None or idx in layer_indices:
            handles.append(blk.attn.register_forward_hook(make_hook(idx)))
    return attn_storage, handles


def unnormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = image.detach().cpu().permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0.0, 1.0)


def attention_grid_to_image(attn_grid: torch.Tensor, height: int, width: int) -> np.ndarray:
    attn_img = F.interpolate(
        attn_grid.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    arr = attn_img.numpy()
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def save_attention_panel(
    image_rgb: np.ndarray,
    overlays: List[np.ndarray],
    titles: List[str],
    out_path: Path,
    suptitle: str,
) -> None:
    ncols = min(4, len(overlays) + 1)
    nrows = int(np.ceil((len(overlays) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    axes[0].imshow(image_rgb)
    axes[0].set_title("original", fontsize=9)
    axes[0].axis("off")
    for ax, overlay, title in zip(axes[1:], overlays, titles):
        ax.imshow(image_rgb)
        ax.imshow(overlay, alpha=0.6, cmap="jet")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(overlays) + 1 :]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def visualize_layer_attention(
    attn_map: torch.Tensor,
    image_tensor: torch.Tensor,
    layer_idx: int,
    output_dir: Path,
    patch_grid_size,
    cam_names: List[str],
    head_indices: Optional[List[int]],
    query_indices: List[int],
    sample_indices: List[int],
) -> None:
    batch_size, num_cams, _channels, height, width = image_tensor.shape
    attn_batch, num_heads, num_tokens, _ = attn_map.shape
    if attn_batch != batch_size * num_cams:
        raise AssertionError(f"Attention batch mismatch: {attn_batch} vs {batch_size * num_cams}")

    num_patches = patch_grid_size[0] * patch_grid_size[1]
    prefix_len = num_tokens - num_patches
    valid_queries = [q for q in query_indices if 0 <= q < prefix_len]
    if not valid_queries:
        raise ValueError(f"No valid query index in {query_indices}; prefix_len={prefix_len}")
    if head_indices is None:
        head_indices = list(range(min(4, num_heads)))

    for sample_idx in sample_indices:
        if sample_idx >= batch_size:
            continue
        for cam_idx in range(min(num_cams, len(cam_names))):
            batch_idx = sample_idx * num_cams + cam_idx
            cam_name = cam_names[cam_idx]
            image_rgb = unnormalize_image(image_tensor[sample_idx, cam_idx])
            cam_dir = output_dir / f"sample_{sample_idx}" / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)

            attn_cam = attn_map[batch_idx]
            attn_mean = attn_cam.mean(dim=0)
            overlays, titles = [], []
            for query_idx in valid_queries:
                patch_attn = attn_mean[query_idx, prefix_len:].reshape(patch_grid_size[0], patch_grid_size[1])
                overlays.append(attention_grid_to_image(patch_attn, height, width))
                titles.append(f"mean q{query_idx}")
            save_attention_panel(
                image_rgb,
                overlays,
                titles,
                cam_dir / f"L{layer_idx:02d}_mean.png",
                f"layer {layer_idx} mean heads",
            )

            for head_idx in head_indices:
                if head_idx >= num_heads:
                    continue
                overlays, titles = [], []
                for query_idx in valid_queries:
                    patch_attn = attn_cam[head_idx, query_idx, prefix_len:].reshape(
                        patch_grid_size[0], patch_grid_size[1]
                    )
                    overlays.append(attention_grid_to_image(patch_attn, height, width))
                    titles.append(f"h{head_idx} q{query_idx}")
                save_attention_panel(
                    image_rgb,
                    overlays,
                    titles,
                    cam_dir / f"L{layer_idx:02d}_H{head_idx}.png",
                    f"layer {layer_idx} head {head_idx}",
                )


def visualize_token_embedding_similarity(agent, real_vit, output_dir: Path) -> None:
    sim_dir = output_dir / "token_embedding_similarity"
    sim_dir.mkdir(parents=True, exist_ok=True)

    scene_embeds = agent._drivor_model.scene_embeds[0, 0].detach().cpu()
    tensors = [scene_embeds]
    labels = [f"scene_{idx}" for idx in range(scene_embeds.shape[0])]

    cls_token = getattr(real_vit, "cls_token", None)
    if cls_token is not None:
        cls = cls_token.detach().cpu().squeeze()
        if cls.dim() == 1:
            cls = cls.unsqueeze(0)
        tensors.append(cls)
        labels.extend([f"cls_{idx}" for idx in range(cls.shape[0])])

    reg_token = getattr(real_vit, "reg_token", None)
    if reg_token is not None:
        reg = reg_token.detach().cpu().squeeze()
        if reg.dim() == 1:
            reg = reg.unsqueeze(0)
        tensors.append(reg)
        labels.extend([f"reg_{idx}" for idx in range(reg.shape[0])])

    all_tokens = torch.cat(tensors, dim=0)
    sim = F.normalize(all_tokens, p=2, dim=-1) @ F.normalize(all_tokens, p=2, dim=-1).T
    sim_np = sim.numpy()

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim_np, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Learnable Token Embedding Cosine Similarity")
    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    fig.tight_layout()
    out_path = sim_dir / "embedding_similarity.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    n_scene = scene_embeds.shape[0]
    if n_scene > 1:
        mask = ~np.eye(n_scene, dtype=bool)
        print(f"Scene token mean pairwise sim: {sim_np[:n_scene, :n_scene][mask].mean():.4f}")
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_training_config(args.config_path, overrides=args.hydra_override)
    if args.data_path:
        set_cfg_value(cfg, "navsim_log_path", args.data_path)
    if args.sensor_blobs_path:
        set_cfg_value(cfg, "sensor_blobs_path", args.sensor_blobs_path)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    agent = instantiate_agent(cfg, args.ckpt_path, device)
    disable_backbone_grid_mask(agent)
    if args.merge_lora:
        merge_lora_to_backbone(agent._drivor_model.image_backbone)

    scene_loader = build_scene_loader(
        data_path=str(cfg.navsim_log_path),
        sensor_blobs_path=str(cfg.sensor_blobs_path),
        split=args.split,
        sensor_config=agent.get_sensor_config(),
        max_scenes=args.max_scenes,
    )
    dataset = build_dataset(cfg, agent, scene_loader, cache_path=None, append_token_to_batch=False)
    dataloader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    batch = next(iter(dataloader))
    features, _targets = batch
    features = move_to_device(features, device)
    image_tensor = first_existing_tensor(features, ["image", "camera_feature"])

    real_vit = get_real_vit_from_backbone(agent._drivor_model.image_backbone)
    layer_indices = parse_int_list(args.layers)
    head_indices = parse_int_list(args.heads)
    query_indices = parse_int_list(args.queries) or [0]
    sample_indices = parse_int_list(args.samples) or [0]

    attn_storage, handles = register_attention_hooks(real_vit, layer_indices=layer_indices)
    with torch.no_grad():
        _ = agent._drivor_model(features)
    for handle in handles:
        handle.remove()

    patch_size = real_vit.patch_embed.patch_size[0]
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]
    patch_grid_size = (height // patch_size, width // patch_size)
    cam_names = get_drivor_camera_names(agent._config)

    print(f"Captured attention from {len(attn_storage)} layers")
    print(f"Patch grid: {patch_grid_size}, cameras: {cam_names}, queries: {query_indices}")

    for layer_idx, attn_map in sorted(attn_storage.items()):
        print(f"Rendering layer {layer_idx}")
        visualize_layer_attention(
            attn_map=attn_map,
            image_tensor=image_tensor.detach().cpu(),
            layer_idx=layer_idx,
            output_dir=output_dir,
            patch_grid_size=patch_grid_size,
            cam_names=cam_names,
            head_indices=head_indices,
            query_indices=query_indices,
            sample_indices=sample_indices,
        )

    visualize_token_embedding_similarity(agent, real_vit, output_dir)
    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
