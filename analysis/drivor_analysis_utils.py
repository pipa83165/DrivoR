#!/usr/bin/env python3
"""Shared helpers for DrivoR analysis scripts.

This module deliberately avoids importing the source project's ``quantize``
package. It follows the current repository's Dataset / vggt_geometry path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "navsim/planning/script/config/training/default_training.yaml"
SCENE_FILTER_ROOT = REPO_ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset


def expand_path(path: Optional[str]) -> Optional[Path]:
    """Expand env vars and user home in a path string."""
    if path is None or path == "":
        return None
    return Path(os.path.expandvars(os.path.expanduser(path)))


def load_training_config(
    config_path: str = str(DEFAULT_TRAINING_CONFIG),
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Compose the current repository's training config with optional overrides."""
    cfg_path = expand_path(config_path)
    if cfg_path is None:
        cfg_path = DEFAULT_TRAINING_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    if cfg_path.suffix in {".yaml", ".yml"}:
        config_dir = cfg_path.parent
        config_name = cfg_path.stem
    else:
        config_dir = cfg_path
        config_name = "default_training"

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    return cfg


def set_cfg_value(cfg: DictConfig, key: str, value) -> None:
    """Update an OmegaConf value even when the config is structured."""
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, key, value, merge=True)


def get_vggt_geometry_cfg(cfg: DictConfig):
    """Return current C1 geometry config, if present."""
    return OmegaConf.select(cfg, "agent.config.vggt_geometry")


def load_scoring_components(config_path: str):
    """Instantiate the PDM simulator/scorer pair from a scoring parameters YAML."""
    path = expand_path(config_path)
    if path is None:
        raise ValueError("config_path is required")
    if not path.is_absolute():
        path = REPO_ROOT / path
    cfg = OmegaConf.load(path)
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise AssertionError("Simulator and scorer proposal sampling must be identical")
    return simulator, scorer


def instantiate_agent(
    cfg: DictConfig,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> AbstractAgent:
    """Instantiate current DrivoR agent and load a Lightning checkpoint."""
    ckpt = str(expand_path(checkpoint_path) or "")
    set_cfg_value(cfg, "agent.checkpoint_path", ckpt)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.to(device)
    agent.eval()
    return agent


def load_scene_filter(
    split: str,
    tokens: Optional[Sequence[str]] = None,
    max_scenes: Optional[int] = None,
    clear_log_names: bool = False,
):
    """Load one of the repository scene_filter YAMLs and optionally restrict tokens."""
    yaml_path = SCENE_FILTER_ROOT / f"{split}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Scene filter not found for split={split}: {yaml_path}")

    cfg = OmegaConf.load(yaml_path)
    scene_filter = instantiate(cfg)
    if tokens is not None:
        scene_filter.tokens = list(tokens)
    if max_scenes is not None and max_scenes > 0:
        scene_filter.max_scenes = int(max_scenes)
    if clear_log_names:
        scene_filter.log_names = None
    return scene_filter


def build_scene_loader(
    data_path: str,
    sensor_blobs_path: str,
    split: str,
    sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
    tokens: Optional[Sequence[str]] = None,
    max_scenes: Optional[int] = None,
    clear_log_names: bool = False,
) -> SceneLoader:
    """Build a SceneLoader using current scene_filter YAMLs."""
    data = expand_path(data_path)
    blobs = expand_path(sensor_blobs_path)
    if data is None:
        raise ValueError("data_path is required")
    if blobs is None:
        blobs = Path("")
    scene_filter = load_scene_filter(
        split=split,
        tokens=tokens,
        max_scenes=max_scenes,
        clear_log_names=clear_log_names,
    )
    return SceneLoader(
        data_path=data,
        sensor_blobs_path=blobs,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )


def build_dataset(
    cfg: DictConfig,
    agent: AbstractAgent,
    scene_loader: SceneLoader,
    cache_path: Optional[str] = None,
    append_token_to_batch: bool = False,
) -> Dataset:
    """Build the current Dataset and preserve vggt_geometry cache behavior."""
    return Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cache_path,
        force_cache_computation=False,
        append_token_to_batch=append_token_to_batch,
        vggt_geometry_cfg=get_vggt_geometry_cfg(cfg),
    )


def build_cache_only_dataset(
    cfg: DictConfig,
    agent: AbstractAgent,
    cache_path: str,
    log_names: Optional[List[str]] = None,
) -> CacheOnlyDataset:
    """Build CacheOnlyDataset with current vggt_geometry support."""
    return CacheOnlyDataset(
        cache_path=cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=log_names,
        vggt_geometry_cfg=get_vggt_geometry_cfg(cfg),
    )


def make_dataloader(
    dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a conservative DataLoader for analysis/inference scripts."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def move_to_device(obj, device: torch.device):
    """Recursively move tensors in a nested batch to the target device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj


def get_drivor_camera_names(agent_config) -> List[str]:
    """Return active camera names in DrivoR image feature order."""
    order = ["cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2"]
    names = []
    for name in order:
        try:
            value = agent_config[name]
        except Exception:
            value = getattr(agent_config, name, [])
        if len(value) > 0:
            names.append(name)
    return names


def get_real_vit_from_backbone(backbone_model):
    """Resolve timm ViT whether the ImgEncoder still wraps it in LoRA or not."""
    vit = backbone_model.model
    return getattr(vit, "lora_vit", vit)


def disable_backbone_grid_mask(agent: AbstractAgent) -> None:
    """Disable training-time grid-mask augmentation for deterministic analysis."""
    model = getattr(agent, "_drivor_model", None)
    backbone = getattr(model, "image_backbone", None)
    if backbone is not None and hasattr(backbone, "use_grid_mask"):
        backbone.use_grid_mask = False


def merge_lora_to_backbone(backbone_model) -> None:
    """Merge current LoRA q/v adapters into the base qkv projection in-place."""
    lora_vit = backbone_model.model
    if not hasattr(lora_vit, "w_As") or not hasattr(lora_vit, "w_Bs"):
        print("[analysis] no LoRA adapters found, skip merge")
        return

    base_vit = lora_vit.lora_vit
    if len(lora_vit.w_As) != len(lora_vit.w_Bs):
        raise AssertionError("LoRA A/B length mismatch")

    lora_layers = getattr(lora_vit, "lora_layer", list(range(len(base_vit.blocks))))
    if len(lora_vit.w_As) != 2 * len(lora_layers):
        raise AssertionError("LoRA adapter count does not match lora_layer")

    merged = 0
    for i, layer_idx in enumerate(lora_layers):
        blk = base_vit.blocks[layer_idx]
        qkv = blk.attn.qkv
        if not hasattr(qkv, "qkv"):
            raise AssertionError(f"block {layer_idx} qkv is not a LoRA wrapper")

        w_a_q = lora_vit.w_As[2 * i]
        w_b_q = lora_vit.w_Bs[2 * i]
        w_a_v = lora_vit.w_As[2 * i + 1]
        w_b_v = lora_vit.w_Bs[2 * i + 1]

        if not isinstance(qkv.layernorm_q, nn.Identity):
            raise AssertionError("layernorm_q is not Identity, cannot merge statically")
        if not isinstance(qkv.layernorm_v, nn.Identity):
            raise AssertionError("layernorm_v is not Identity, cannot merge statically")

        dim = qkv.qkv.in_features
        delta_q = (w_b_q.weight @ w_a_q.weight).to(qkv.qkv.weight.device, qkv.qkv.weight.dtype)
        delta_v = (w_b_v.weight @ w_a_v.weight).to(qkv.qkv.weight.device, qkv.qkv.weight.dtype)

        with torch.no_grad():
            qkv.qkv.weight[:dim, :] += delta_q
            qkv.qkv.weight[-dim:, :] += delta_v
            blk.attn.qkv = qkv.qkv

        merged += 1

    backbone_model.model = base_vit
    backbone_model.use_lora = False
    print(f"[analysis] LoRA merge complete ({merged} blocks)")


def build_image_backbone_config(
    agent_config_path: str,
    num_scene_tokens: Optional[int] = None,
) -> DictConfig:
    """Build ImgEncoder config from the current DrivoR agent YAML."""
    path = expand_path(agent_config_path)
    if path is None:
        raise ValueError("agent_config_path is required")
    if not path.is_absolute():
        path = REPO_ROOT / path
    cfg = OmegaConf.load(path)
    image_cfg = OmegaConf.create(OmegaConf.to_container(cfg.config.image_backbone, resolve=True))
    image_cfg.image_size = list(cfg.config.image_size)
    image_cfg.num_scene_tokens = int(num_scene_tokens or cfg.config.num_scene_tokens)
    image_cfg.tf_d_model = int(cfg.config.tf_d_model)
    if "in_chans" not in image_cfg:
        image_cfg.in_chans = 3
    return image_cfg


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated or bracket-like integer list CLI values."""
    if value is None or value == "":
        return None
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if text.strip() == "":
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def first_existing_tensor(features: Dict, keys: Iterable[str]):
    """Return the first tensor-like feature for any of the given keys."""
    for key in keys:
        if key in features:
            return features[key]
    raise KeyError(f"None of these feature keys were found: {list(keys)}")
