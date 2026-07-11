#!/usr/bin/env python3
"""Export scene-level DrivoR proposal generation and ranking diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
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
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score


SUB_SCORE_COLS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True, help="DrivoR Lightning checkpoint")
    parser.add_argument("--metric_cache_path", required=True)
    parser.add_argument(
        "--scoring_config",
        default="navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml",
    )
    parser.add_argument(
        "--config_path",
        default="navsim/planning/script/config/training/default_training.yaml",
    )
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--sensor_blobs_path", default=None)
    parser.add_argument("--output_dir", default="analysis_output/proposal_diagnostics")
    parser.add_argument("--num_scenes", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hydra_override", action="append", default=[])
    return parser.parse_args()


def load_scoring_components(config_path: str):
    config = OmegaConf.load(config_path)
    simulator = instantiate(config.simulator)
    scorer = instantiate(config.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise AssertionError("Simulator and scorer proposal sampling must be identical")
    return simulator, scorer


def score_proposals(
    tokens: List[str], proposals: torch.Tensor, metric_cache_loader, simulator, scorer
) -> np.ndarray:
    missing = [token for token in tokens if token not in metric_cache_loader.metric_cache_paths]
    if missing:
        raise KeyError(f"Missing metric cache for {len(missing)} token(s), first={missing[0]}")

    batch_scores = []
    for token, proposal_set in zip(tokens, proposals.detach().cpu().numpy()):
        metric_cache = metric_cache_loader.get_from_token(token)
        proposal_scores = []
        for proposal in proposal_set:
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(proposal.astype(np.float32)),
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            result_dict = asdict(result)
            proposal_scores.append(
                [result_dict[column] for column in SUB_SCORE_COLS + ["score"]]
            )
        batch_scores.append(proposal_scores)
    return np.asarray(batch_scores, dtype=np.float64)


def build_diagnostic_rows(
    tokens: List[str], predicted_scores: np.ndarray, true_scores: np.ndarray
) -> tuple[List[Dict], List[Dict]]:
    if predicted_scores.shape != true_scores[..., -1].shape:
        raise ValueError(
            "Predicted/true proposal score shapes differ: "
            f"{predicted_scores.shape} vs {true_scores[..., -1].shape}"
        )

    scene_rows: List[Dict] = []
    proposal_rows: List[Dict] = []
    for batch_idx, token in enumerate(tokens):
        predicted = predicted_scores[batch_idx]
        actual = true_scores[batch_idx, :, -1]
        selected_idx = int(np.argmax(predicted))
        oracle_idx = int(np.argmax(actual))
        selected_score = float(actual[selected_idx])
        oracle_score = float(actual[oracle_idx])
        top_k = min(5, len(actual))
        true_top_k = np.argpartition(actual, -top_k)[-top_k:]

        row = {
            "token": token,
            "valid": True,
            "num_proposals": int(len(actual)),
            "selected_idx": selected_idx,
            "oracle_idx": oracle_idx,
            "selected_score": selected_score,
            "oracle_score": oracle_score,
            "ranking_regret": oracle_score - selected_score,
            "hit_at_1": selected_idx == oracle_idx,
            "hit_at_5": selected_idx in true_top_k,
            "selected_predicted_score": float(predicted[selected_idx]),
            "oracle_predicted_score": float(predicted[oracle_idx]),
        }
        for metric_idx, metric in enumerate(SUB_SCORE_COLS):
            row[f"selected_{metric}"] = float(true_scores[batch_idx, selected_idx, metric_idx])
            row[f"oracle_{metric}"] = float(true_scores[batch_idx, oracle_idx, metric_idx])
        scene_rows.append(row)

        for proposal_idx in range(len(actual)):
            proposal_row = {
                "token": token,
                "proposal_idx": proposal_idx,
                "predicted_score": float(predicted[proposal_idx]),
                "true_score": float(actual[proposal_idx]),
                "is_selected": proposal_idx == selected_idx,
                "is_oracle": proposal_idx == oracle_idx,
            }
            for metric_idx, metric in enumerate(SUB_SCORE_COLS):
                proposal_row[metric] = float(true_scores[batch_idx, proposal_idx, metric_idx])
            proposal_rows.append(proposal_row)
    return scene_rows, proposal_rows


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

    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
    simulator, scorer = load_scoring_components(args.scoring_config)
    scene_loader = build_scene_loader(
        data_path=str(cfg.navsim_log_path),
        sensor_blobs_path=str(cfg.sensor_blobs_path),
        split=args.split,
        sensor_config=agent.get_sensor_config(),
        max_scenes=args.num_scenes if args.num_scenes > 0 else None,
    )
    dataset = build_dataset(cfg, agent, scene_loader, cache_path=None, append_token_to_batch=True)
    dataloader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    scene_rows: List[Dict] = []
    proposal_rows: List[Dict] = []
    for features, _targets, tokens in tqdm(dataloader, desc="Scoring proposal sets"):
        tokens = [str(token) for token in tokens]
        features = move_to_device(features, device)
        with torch.no_grad():
            predictions = agent.forward(features)
        proposals = predictions["proposals"]
        predicted_scores = predictions["pdm_score"].detach().float().cpu().numpy()
        true_scores = score_proposals(tokens, proposals, metric_cache_loader, simulator, scorer)
        batch_scene_rows, batch_proposal_rows = build_diagnostic_rows(
            tokens, predicted_scores, true_scores
        )
        scene_rows.extend(batch_scene_rows)
        proposal_rows.extend(batch_proposal_rows)

    if not scene_rows:
        raise RuntimeError("No scenes were exported")
    scene_df = pd.DataFrame(scene_rows)
    proposal_df = pd.DataFrame(proposal_rows)
    scene_df.to_csv(output_dir / "proposal_diagnostics.csv", index=False)
    proposal_df.to_csv(output_dir / "proposal_scores.csv", index=False)

    print(f"Exported {len(scene_df)} scenes and {len(proposal_df)} proposals to {output_dir}")
    print(
        f"selected={scene_df['selected_score'].mean():.4f}, "
        f"oracle={scene_df['oracle_score'].mean():.4f}, "
        f"regret={scene_df['ranking_regret'].mean():.4f}, "
        f"hit@1={scene_df['hit_at_1'].mean():.4f}, hit@5={scene_df['hit_at_5'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
