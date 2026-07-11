#!/usr/bin/env python3
"""Visualize low-score NAVSIM cases with GT and DrivoR predictions in BEV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from drivor_analysis_utils import (
    build_dataset,
    build_scene_loader,
    disable_backbone_grid_mask,
    instantiate_agent,
    load_training_config,
    make_dataloader,
    move_to_device,
    set_cfg_value,
)
from navsim.common.dataclasses import Trajectory
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize low-score cases with BEV overlays")
    parser.add_argument("--token_list", type=str, required=True, help="One scenario token per line")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Lightning checkpoint")
    parser.add_argument(
        "--config_path",
        type=str,
        default="navsim/planning/script/config/training/default_training.yaml",
    )
    parser.add_argument("--split", type=str, default="navtest")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sensor_blobs_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./analysis_output/low_score_viz")
    parser.add_argument("--num_scenes", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tokens_per_figure", type=int, default=10)
    parser.add_argument("--proposal_index", type=int, default=-1, help="-1 uses predictions['trajectory']; otherwise proposals[:, idx]")
    parser.add_argument("--keep_split_log_names", action="store_true", help="Do not clear split log_names when filtering tokens")
    parser.add_argument("--hydra_override", action="append", default=[], help="Extra Hydra override, repeatable")
    return parser.parse_args()


def load_tokens(token_list_path: str, num_scenes: int) -> List[str]:
    with open(token_list_path, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    if num_scenes > 0:
        tokens = tokens[:num_scenes]
    if not tokens:
        raise ValueError(f"No tokens loaded from {token_list_path}")
    return tokens


def _draw_single_bev_on_ax(ax, scene, gt_traj, pred_traj, title: str = "") -> None:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    if gt_traj is not None:
        add_trajectory_to_bev_ax(ax, gt_traj, TRAJECTORY_CONFIG["human"])
    if pred_traj is not None:
        add_trajectory_to_bev_ax(ax, pred_traj, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)
    if title:
        ax.set_title(title, fontsize=7)


def plot_bev_grid(results: List[Dict], output_dir: Path, tokens_per_figure: int) -> None:
    if not results:
        print("No visualization results to plot")
        return

    ncols = min(5, tokens_per_figure)
    nrows = (tokens_per_figure + ncols - 1) // ncols
    subplot_size = 2.6

    for batch_start in range(0, len(results), tokens_per_figure):
        batch = results[batch_start : batch_start + tokens_per_figure]
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * subplot_size, nrows * subplot_size))
        axes = np.array(axes).reshape(-1)
        for idx, ax in enumerate(axes):
            if idx < len(batch):
                rec = batch[idx]
                _draw_single_bev_on_ax(
                    ax,
                    rec["scene"],
                    rec["gt_traj"],
                    rec["pred_traj"],
                    title=rec["token"][:12],
                )
            else:
                ax.axis("off")
        fig.tight_layout(pad=0.5)
        save_path = output_dir / f"batch_{batch_start // tokens_per_figure:03d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")


def run_predictions(agent, dataloader, scene_loader, device: torch.device, proposal_index: int) -> List[Dict]:
    results: List[Dict] = []
    num_gt_poses = int(agent._config.trajectory_sampling.num_poses)

    for batch in tqdm(dataloader, desc="Inferencing"):
        features, _targets, tokens = batch
        features = move_to_device(features, device)
        with torch.no_grad():
            predictions = agent.forward(features)
            if proposal_index >= 0:
                pred_poses = predictions["proposals"][:, proposal_index]
            else:
                pred_poses = predictions["trajectory"]

        for idx, token in enumerate(tokens):
            token = str(token)
            scene = scene_loader.get_scene_from_token(token)
            try:
                gt_traj = scene.get_future_trajectory(num_trajectory_frames=num_gt_poses)
            except Exception:
                gt_traj = None
            pred_np = pred_poses[idx].detach().cpu().numpy().astype(np.float32)
            pred_traj = Trajectory(pred_np)
            results.append({"token": token, "scene": scene, "gt_traj": gt_traj, "pred_traj": pred_traj})
    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_training_config(args.config_path, overrides=args.hydra_override)
    if args.data_path:
        set_cfg_value(cfg, "navsim_log_path", args.data_path)
    if args.sensor_blobs_path:
        set_cfg_value(cfg, "sensor_blobs_path", args.sensor_blobs_path)

    device = torch.device(args.device)
    agent = instantiate_agent(cfg, args.ckpt_path, device)
    disable_backbone_grid_mask(agent)

    tokens = load_tokens(args.token_list, args.num_scenes)
    scene_loader = build_scene_loader(
        data_path=str(cfg.navsim_log_path),
        sensor_blobs_path=str(cfg.sensor_blobs_path),
        split=args.split,
        sensor_config=agent.get_sensor_config(),
        tokens=tokens,
        clear_log_names=not args.keep_split_log_names,
    )
    available = set(scene_loader.tokens)
    filtered_tokens = [token for token in tokens if token in available]
    if len(filtered_tokens) != len(tokens):
        print(f"Filtered unavailable tokens: kept {len(filtered_tokens)} / {len(tokens)}")
    if not filtered_tokens:
        raise RuntimeError("None of the requested tokens are available in the scene loader")

    dataset = build_dataset(cfg, agent, scene_loader, cache_path=None, append_token_to_batch=True)
    dataloader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    results = run_predictions(agent, dataloader, scene_loader, device, args.proposal_index)
    plot_bev_grid(results, output_dir, args.tokens_per_figure)
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
