#!/usr/bin/env python3
"""Extract reproducible scene attributes for paired score slicing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from drivor_analysis_utils import build_scene_loader
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from tqdm import tqdm

from navsim.common.dataclasses import Scene, SensorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--sensor_blobs_path", default="")
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--token_list", default=None, help="Optional one-token-per-line filter")
    parser.add_argument("--num_scenes", type=int, default=-1)
    parser.add_argument("--output_csv", default="analysis_output/scene_attributes.csv")
    parser.add_argument("--agent_radius", type=float, default=30.0)
    parser.add_argument("--intersection_radius", type=float, default=20.0)
    return parser.parse_args()


def load_tokens(path: Optional[str]) -> Optional[list[str]]:
    if not path:
        return None
    with open(path, "r") as file:
        tokens = [line.strip() for line in file if line.strip()]
    if not tokens:
        raise ValueError(f"No tokens found in {path}")
    return tokens


def interval_label(value: float, edges: list[float], labels: list[str]) -> str:
    if len(labels) != len(edges) + 1:
        raise ValueError("labels must have exactly len(edges) + 1 entries")
    return labels[int(np.digitize(value, edges, right=False))]


def maneuver_label(distance: float, final_heading: float) -> str:
    if distance < 2.0:
        return "stationary"
    if abs(final_heading) < 0.20:
        return "straight"
    if final_heading >= 0.35:
        return "left"
    if final_heading <= -0.35:
        return "right"
    return "curved_left" if final_heading > 0 else "curved_right"


def map_attributes(scene: Scene, intersection_radius: float) -> tuple[bool, bool]:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    ego_pose = scene.frames[frame_idx].ego_status.ego_pose
    point = Point2D(float(ego_pose[0]), float(ego_pose[1]))
    try:
        in_intersection = scene.map_api.is_in_layer(point, SemanticMapLayer.INTERSECTION)
        nearby = scene.map_api.get_proximal_map_objects(
            point, intersection_radius, [SemanticMapLayer.INTERSECTION]
        )
        near_intersection = len(nearby[SemanticMapLayer.INTERSECTION]) > 0
    except (AttributeError, KeyError, NotImplementedError, RuntimeError, AssertionError, ValueError):
        in_intersection = False
        near_intersection = False
    return bool(in_intersection), bool(near_intersection)


def scene_attributes(scene: Scene, agent_radius: float, intersection_radius: float) -> Dict:
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]
    trajectory = scene.get_future_trajectory().poses
    xy = trajectory[:, :2]
    segment_lengths = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0), axis=1)
    path_length = float(segment_lengths.sum())
    final_heading = float(np.arctan2(np.sin(trajectory[-1, 2]), np.cos(trajectory[-1, 2])))
    heading_change = float(np.max(np.abs(np.arctan2(np.sin(trajectory[:, 2]), np.cos(trajectory[:, 2])))))

    speed = float(np.linalg.norm(frame.ego_status.ego_velocity[:2]))
    acceleration = float(np.linalg.norm(frame.ego_status.ego_acceleration[:2]))
    boxes = np.asarray(frame.annotations.boxes)
    names = np.asarray(frame.annotations.names, dtype=object)
    dynamic_names = {"vehicle", "pedestrian", "bicycle"}
    if len(boxes):
        distances = np.linalg.norm(boxes[:, :2], axis=1)
        dynamic_mask = np.asarray([name in dynamic_names for name in names])
        nearby_mask = dynamic_mask & (distances <= agent_radius)
        nearby_names = names[nearby_mask]
        nearest_agent_distance = float(distances[dynamic_mask].min()) if dynamic_mask.any() else np.inf
    else:
        nearby_names = np.asarray([], dtype=object)
        nearest_agent_distance = np.inf

    dynamic_count = len(nearby_names)
    vru_count = int(sum(name in {"pedestrian", "bicycle"} for name in nearby_names))
    in_intersection, near_intersection = map_attributes(scene, intersection_radius)
    red_light_count = int(sum(bool(is_red) for _lane_id, is_red in frame.traffic_lights))

    return {
        "token": scene.scene_metadata.initial_token,
        "log_name": scene.scene_metadata.log_name,
        "map_name": scene.scene_metadata.map_name,
        "maneuver": maneuver_label(path_length, final_heading),
        "path_length_m": path_length,
        "final_lateral_displacement_m": float(trajectory[-1, 1]),
        "final_heading_change_rad": final_heading,
        "max_heading_change_rad": heading_change,
        "geometry_complexity": interval_label(
            heading_change, [0.10, 0.35], ["low", "medium", "high"]
        ),
        "ego_speed_mps": speed,
        "speed_bin": interval_label(speed, [2.0, 5.0, 10.0], ["0-2", "2-5", "5-10", "10+"]),
        "ego_acceleration_mps2": acceleration,
        "in_intersection": in_intersection,
        "near_intersection": near_intersection,
        "has_traffic_light": len(frame.traffic_lights) > 0,
        "has_red_light": red_light_count > 0,
        "red_light_count": red_light_count,
        "agent_radius_m": agent_radius,
        "dynamic_agent_count": dynamic_count,
        "agent_density_bin": interval_label(
            dynamic_count, [1, 5, 10], ["0", "1-4", "5-9", "10+"]
        ),
        "nearest_agent_distance_m": nearest_agent_distance,
        "nearest_agent_distance_bin": interval_label(
            nearest_agent_distance, [5.0, 15.0, 30.0], ["0-5", "5-15", "15-30", "30+/none"]
        ),
        "vru_count": vru_count,
        "has_vru": vru_count > 0,
    }


def main() -> None:
    args = parse_args()
    tokens = load_tokens(args.token_list)
    scene_loader = build_scene_loader(
        data_path=args.data_path,
        sensor_blobs_path=args.sensor_blobs_path,
        split=args.split,
        sensor_config=SensorConfig.build_no_sensors(),
        tokens=tokens,
        max_scenes=args.num_scenes if args.num_scenes > 0 else None,
        clear_log_names=tokens is not None,
    )
    rows = [
        scene_attributes(
            scene_loader.get_scene_from_token(token),
            args.agent_radius,
            args.intersection_radius,
        )
        for token in tqdm(scene_loader.tokens, desc="Extracting scene attributes")
    ]
    if not rows:
        raise RuntimeError("No scenes were available for attribute extraction")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} scene attribute rows to {output_path}")


if __name__ == "__main__":
    main()
