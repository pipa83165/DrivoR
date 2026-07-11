#!/usr/bin/env python3
"""Evaluate human GT trajectories with NAVSIM PDM-Score."""

from __future__ import annotations

import argparse
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from tqdm import tqdm

from drivor_analysis_utils import build_scene_loader, load_scoring_components
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GT trajectories with PDM-Score")
    parser.add_argument("--metric_cache_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sensor_blobs_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./analysis_output/gt_score_analysis")
    parser.add_argument("--num_scenes", type=int, default=-1)
    parser.add_argument("--split", type=str, default="navtest")
    parser.add_argument(
        "--config_path",
        type=str,
        default="navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator, scorer = load_scoring_components(args.config_path)
    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
    print(f"Metric cache tokens: {len(metric_cache_loader.tokens)}")

    scene_loader = build_scene_loader(
        data_path=args.data_path,
        sensor_blobs_path=args.sensor_blobs_path,
        split=args.split,
        sensor_config=SensorConfig.build_no_sensors(),
        max_scenes=args.num_scenes if args.num_scenes > 0 else None,
    )
    print(f"Scene loader tokens: {len(scene_loader.tokens)}")

    metric_tokens = set(metric_cache_loader.tokens)
    tokens_to_evaluate = [token for token in scene_loader.tokens if token in metric_tokens]
    if args.num_scenes > 0:
        tokens_to_evaluate = tokens_to_evaluate[: args.num_scenes]
    print(f"Evaluating overlap tokens: {len(tokens_to_evaluate)}")

    pdm_results: List[Dict] = []
    for token in tqdm(tokens_to_evaluate, desc="Scoring GT"):
        score_row: Dict = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            gt_trajectory = scene.get_future_trajectory()

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=gt_trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            print(f"Failed for token {token}: {e}")
            traceback.print_exc()
            score_row["valid"] = False
        pdm_results.append(score_row)

    df = pd.DataFrame(pdm_results)
    if len(df) == 0:
        raise RuntimeError("No GT trajectories were evaluated")

    valid_df = df[df["valid"].astype(bool)]
    if len(valid_df) > 0:
        avg_row = valid_df.drop(columns=["token", "valid"]).mean(skipna=True)
        avg_row["token"] = "average"
        avg_row["valid"] = True
        df.loc[len(df)] = avg_row

    csv_path = output_dir / "gt_pdm_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    if len(valid_df) > 0:
        print("\n=== GT Trajectory PDM Score Statistics ===")
        score_cols = [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
            "score",
        ]
        for col in score_cols:
            vals = valid_df[col].dropna()
            print(
                f"  {col:40s}: mean={vals.mean():.4f}, median={vals.median():.4f}, "
                f"min={vals.min():.4f}, max={vals.max():.4f}, "
                f"zero={(vals == 0).sum():4d} / {len(vals)}"
            )
        print(f"\n  Valid scenes: {len(valid_df)} / {len(df) - 1}")


if __name__ == "__main__":
    main()
