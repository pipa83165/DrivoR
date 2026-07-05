import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from tqdm import tqdm

from navsim.agents.drivoR.vggt_geometry import (
    CACHE_DTYPE,
    VGGT_GEOMETRY_SCENE_DICT_KEYS,
    METADATA_FILENAME,
    STATS_FILENAME,
    TOKEN_INDEX_FILENAME,
    FrozenVggtGeometryTeacher,
    build_fingerprint,
    file_sha256,
    vggt_geometry_cache_file,
    tokens_per_camera,
)
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_CAMERA_ORDER = VGGT_GEOMETRY_SCENE_DICT_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen VGGT-Omega geometry register tokens.")
    parser.add_argument("--checkpoint", default="weights/vggt_omega_1b_512.pt", help="VGGT-Omega checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Output token cache directory.")
    parser.add_argument("--config-dir", default="navsim/planning/script/config/training", help="Hydra training config directory.")
    parser.add_argument("--config-name", default="default_training", help="Hydra training config name.")
    parser.add_argument("--split", default="train", help="train/navtrain, val/navval, or trainval.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tokens that already have a cache file.")
    parser.add_argument("--validate-existing", action="store_true", help="Deep-check existing token files before skipping.")
    parser.add_argument("--preprocess-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--independent-forward", action="store_true", help="Run each camera independently instead of joint 4-camera forward.")
    parser.add_argument("--forward-mode", choices=("joint", "independent"), default=None, help="Backward-compatible alias.")
    parser.add_argument("--use-camera-token", action="store_true")
    parser.add_argument("--device", default="cuda", help="Torch device for VGGT-Omega inference.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-finalize", action="store_true", help="Do not rewrite metadata/token_index/noise_stats after this shard.")
    parser.add_argument("overrides", nargs="*", help="Optional Hydra overrides, e.g. train_test_split=navmini")
    return parser.parse_args()


def joint_forward_from_args(args: argparse.Namespace) -> bool:
    if args.forward_mode is not None:
        return args.forward_mode == "joint"
    return not args.independent_forward


def cache_cfg_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "checkpoint_path": args.checkpoint,
        "vggt_dim": 2048,
        "num_registers": 16,
        "use_camera_token": bool(args.use_camera_token),
        "joint_forward": joint_forward_from_args(args),
        "preprocess_mode": args.preprocess_mode,
        "image_resolution": int(args.image_resolution),
    }


def load_cfg(args: argparse.Namespace):
    config_dir = str(Path(args.config_dir).resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=args.config_name, overrides=list(args.overrides))


def build_scene_loader(cfg, args: argparse.Namespace) -> SceneLoader:
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    split = args.split.lower()
    selected_logs: List[str] = []
    if split in ("train", "navtrain", "trainval"):
        selected_logs.extend(cfg.train_logs)
    if split in ("val", "navval", "trainval"):
        selected_logs.extend(cfg.val_logs)
    if not selected_logs:
        raise ValueError("--split must be train/navtrain, val/navval, or trainval")

    if scene_filter.log_names is not None:
        scene_filter.log_names = [log_name for log_name in scene_filter.log_names if log_name in selected_logs]
    else:
        scene_filter.log_names = selected_logs

    return SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )


def iter_tokens(loader: SceneLoader, args: argparse.Namespace) -> Iterable[Tuple[str, List[Dict]]]:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    tokens = list(loader.tokens)
    if args.max_samples is not None:
        tokens = tokens[: args.max_samples]
    for index, token in enumerate(tokens):
        if index % args.num_shards == args.shard_index:
            yield token, loader.scene_frames_dicts[token]


def camera_paths(
    scene_dict_list: List[Dict],
    sensor_blobs_path: Path,
    camera_order: Sequence[str],
    num_history_frames: int,
) -> List[Path]:
    frame = scene_dict_list[num_history_frames - 1]
    paths = []
    for camera_name in camera_order:
        if camera_name not in frame["cams"]:
            raise KeyError(f"Camera {camera_name} is missing for token {frame['token']}")
        paths.append(sensor_blobs_path / frame["cams"][camera_name]["data_path"])
    return paths


def extract_tokens(teacher: FrozenVggtGeometryTeacher, images: torch.Tensor) -> torch.Tensor:
    device = next(teacher.parameters()).device
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    tokens = teacher(images.unsqueeze(0))[0]
    return tokens.detach().to(device="cpu", dtype=CACHE_DTYPE).contiguous()


def select_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, dict):
        for key in ("tokens", "vggt_tokens", "geo_tokens", "registers", "camera_and_register_tokens"):
            if key in data:
                return select_tensor(data[key])
    raise TypeError(f"Could not find token tensor in {type(data)!r}")


def validate_token_file(path: Path, expected_shape: Tuple[int, int, int]) -> None:
    tensor = select_tensor(torch.load(path, map_location="cpu"))
    if tuple(tensor.shape) != expected_shape or tensor.dtype != CACHE_DTYPE:
        raise ValueError(f"Bad cache file {path}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def cached_token_paths(output_dir: Path) -> Dict[str, Path]:
    paths = {}
    for path in output_dir.rglob("*.pt"):
        if path.name == STATS_FILENAME or ".tmp." in path.name:
            continue
        paths[path.stem] = path
    return dict(sorted(paths.items()))


def compute_noise_stats(paths: Sequence[Path], vggt_dim: int) -> Dict[str, torch.Tensor]:
    count = 0
    mean = torch.zeros(vggt_dim, dtype=torch.float64)
    m2 = torch.zeros(vggt_dim, dtype=torch.float64)

    for path in tqdm(paths, desc="Computing VGGT geometry noise stats"):
        token = select_tensor(torch.load(path, map_location="cpu")).float().reshape(-1, vggt_dim).double()
        batch_count = token.shape[0]
        batch_mean = token.mean(dim=0)
        batch_m2 = ((token - batch_mean) ** 2).sum(dim=0)

        if count == 0:
            mean = batch_mean
            m2 = batch_m2
            count = batch_count
            continue

        delta = batch_mean - mean
        total = count + batch_count
        mean = mean + delta * batch_count / total
        m2 = m2 + batch_m2 + delta.pow(2) * count * batch_count / total
        count = total

    if count < 2:
        std = torch.ones(vggt_dim, dtype=torch.float32)
    else:
        std = torch.sqrt(m2 / (count - 1)).float().clamp_min(1e-6)
    return {"mean": mean.float(), "std": std, "count": torch.tensor(count)}


def finalize_cache(output_dir: Path, cache_cfg: Dict[str, object], script_path: Path) -> None:
    paths = cached_token_paths(output_dir)
    if not paths:
        raise RuntimeError(f"No VGGT geometry token .pt files found in {output_dir}")

    rel_index = {token: str(path.relative_to(output_dir)) for token, path in paths.items()}
    atomic_write_json(rel_index, output_dir / TOKEN_INDEX_FILENAME)

    metadata = build_fingerprint(
        cache_cfg,
        ckpt_sha256=file_sha256(str(cache_cfg["checkpoint_path"])),
        cache_script_path=script_path,
    )
    metadata["num_cached_samples"] = len(paths)
    atomic_write_json(metadata, output_dir / METADATA_FILENAME)

    stats = compute_noise_stats(list(paths.values()), int(cache_cfg["vggt_dim"]))
    atomic_torch_save(stats, output_dir / STATS_FILENAME)


def main() -> None:
    args = parse_args()
    cache_cfg = cache_cfg_from_args(args)
    cfg = load_cfg(args)
    loader = build_scene_loader(cfg, args)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("VGGT-Omega cache generation expects CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    teacher = FrozenVggtGeometryTeacher(
        args.checkpoint,
        use_camera_token=args.use_camera_token,
        joint_forward=joint_forward_from_args(args),
    ).to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_shape = (4, tokens_per_camera(cache_cfg), int(cache_cfg["vggt_dim"]))
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    num_history_frames = cfg.train_test_split.scene_filter.num_history_frames

    written = 0
    skipped = 0
    for token, scene_dict_list in tqdm(list(iter_tokens(loader, args)), desc="Caching VGGT-Omega geometry tokens"):
        output_path = vggt_geometry_cache_file(output_dir, token)
        if args.skip_existing and output_path.is_file():
            if args.validate_existing:
                validate_token_file(output_path, expected_shape)
            skipped += 1
            continue

        paths = camera_paths(scene_dict_list, sensor_blobs_path, DEFAULT_CAMERA_ORDER, num_history_frames)
        images = load_and_preprocess_images(
            [str(path) for path in paths],
            mode=args.preprocess_mode,
            image_resolution=args.image_resolution,
            patch_size=args.patch_size,
        )
        tokens = extract_tokens(teacher, images)
        if tuple(tokens.shape) != expected_shape:
            raise RuntimeError(f"Unexpected token shape for {token}: {tuple(tokens.shape)} != {expected_shape}")
        atomic_torch_save(tokens, output_path)
        written += 1

    if not args.no_finalize:
        finalize_cache(output_dir, cache_cfg, Path(__file__))

    print(f"VGGT geometry cache done: written={written} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()
