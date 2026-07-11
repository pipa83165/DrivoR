import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from navsim.agents.drivoR.timm_layers import LayerScale


logger = logging.getLogger(__name__)

VGGT_GEOMETRY_CAMERA_ORDER = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")
VGGT_GEOMETRY_SCENE_DICT_KEYS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")

CACHE_DTYPE = torch.float16
METADATA_FILENAME = "metadata.json"
TOKEN_INDEX_FILENAME = "token_index.json"
LEGACY_INDEX_FILENAME = "index.json"
STATS_FILENAME = "noise_stats.pt"

STRICT_KEYS = (
    "checkpoint_name",
    "checkpoint_sha256",
    "vggt_dim",
    "vggt_dim_semantics",
    "num_registers",
    "tokens_per_camera",
    "camera_order",
    "preprocess",
    "joint_forward",
    "use_camera_token",
    "cache_dtype",
)


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def tokens_per_camera(cfg: Any) -> int:
    num_registers = int(cfg_get(cfg, "num_registers", cfg_get(cfg, "tokens_per_camera", 16)))
    use_camera_token = bool(cfg_get(cfg, "use_camera_token", cfg_get(cfg, "include_camera_token", False)))
    return num_registers + int(use_camera_token)


def cache_dir_from_cfg(cfg: Any) -> Path:
    cache_dir = cfg_get(cfg, "cache_dir", cfg_get(cfg, "cache_path", None))
    if cache_dir in (None, ""):
        raise ValueError("VGGT geometry is enabled with source=cache, but vggt_geometry.cache_dir is empty")
    return Path(str(cache_dir)).expanduser()


@lru_cache(maxsize=16)
def file_sha256(path: Union[str, Path]) -> str:
    sha = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_fingerprint(cfg: Any, ckpt_sha256: str, cache_script_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    import vggt_omega.utils.load_fn as load_fn_mod

    return {
        "checkpoint_name": Path(str(cfg_get(cfg, "checkpoint_path"))).name,
        "checkpoint_sha256": ckpt_sha256,
        "vggt_dim": int(cfg_get(cfg, "vggt_dim", 2048)),
        "vggt_dim_semantics": "cat(frame_attn_1024, global_attn_1024)",
        "num_registers": int(cfg_get(cfg, "num_registers", cfg_get(cfg, "tokens_per_camera", 16))),
        "tokens_per_camera": tokens_per_camera(cfg),
        "camera_order": list(VGGT_GEOMETRY_CAMERA_ORDER),
        "preprocess": {
            "load_fn": "vggt_omega.utils.load_fn.load_and_preprocess_images",
            "load_fn_sha256": file_sha256(load_fn_mod.__file__),
            "mode": str(cfg_get(cfg, "preprocess_mode", "balanced")),
            "image_resolution": int(cfg_get(cfg, "image_resolution", 512)),
            "color_aug": False,
        },
        "joint_forward": bool(cfg_get(cfg, "joint_forward", True)),
        "use_camera_token": bool(cfg_get(cfg, "use_camera_token", False)),
        "cache_dtype": "float16",
        "cache_script_sha256": file_sha256(cache_script_path) if cache_script_path else None,
    }


def build_fingerprint_from_cfg(cfg: Any) -> Dict[str, Any]:
    return build_fingerprint(cfg, ckpt_sha256=file_sha256(cfg_get(cfg, "checkpoint_path")))


def validate_fingerprint(expected: Dict[str, Any], cache_dir: Path, force_ignore: bool = False) -> None:
    metadata_path = cache_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        msg = f"VGGT geometry cache metadata is missing: {metadata_path}"
        if force_ignore:
            logger.warning("FORCE-IGNORE-FINGERPRINT: %s", msg)
            return
        raise FileNotFoundError(msg)

    metadata = json.loads(metadata_path.read_text())
    mismatches = {key: (expected.get(key), metadata.get(key)) for key in STRICT_KEYS if metadata.get(key) != expected.get(key)}
    if mismatches:
        msg = f"VGGT geometry cache fingerprint mismatch at {cache_dir}: {mismatches}"
        if force_ignore:
            logger.warning("FORCE-IGNORE-FINGERPRINT: %s", msg)
        else:
            raise RuntimeError(msg)

    soft_mismatches = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if key not in STRICT_KEYS and metadata.get(key) != value
    }
    if soft_mismatches:
        logger.warning("VGGT geometry cache non-strict metadata differs: %s", soft_mismatches)


def vggt_geometry_cache_file(cache_dir: Path, token: str) -> Path:
    return cache_dir / token[:2] / f"{token}.pt"


def _load_token_index(cache_dir: Path) -> Optional[Dict[str, str]]:
    token_index = cache_dir / TOKEN_INDEX_FILENAME
    if token_index.is_file():
        tokens = json.loads(token_index.read_text())
        if isinstance(tokens, list):
            return {str(token): str(vggt_geometry_cache_file(cache_dir, str(token)).relative_to(cache_dir)) for token in tokens}
        if isinstance(tokens, dict):
            return {str(token): str(path) for token, path in tokens.items()}

    legacy_index = cache_dir / LEGACY_INDEX_FILENAME
    if legacy_index.is_file():
        index = json.loads(legacy_index.read_text())
        index = index.get("tokens", index)
        if isinstance(index, dict):
            return {str(token): str(path) for token, path in index.items()}
    return None


def _select_tensor(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Mapping):
        for key in ("tokens", "vggt_tokens", "geo_tokens", "registers", "camera_and_register_tokens"):
            if key in data:
                return _select_tensor(data[key])
    raise TypeError(f"Could not find a tensor in cached VGGT geometry token object of type {type(data)!r}")


class VggtGeometryProjector(nn.Module):
    """Project frozen VGGT-Omega tokens into DrivoR decoder memory space."""

    def __init__(
        self,
        vggt_dim: int,
        d_model: int,
        num_cams: int = 4,
        tokens_per_cam: int = 16,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.tokens_per_cam = int(tokens_per_cam)
        self.input_ln = nn.LayerNorm(vggt_dim)
        self.proj = nn.Linear(vggt_dim, d_model)
        self.branch_embed = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.cam_embed = nn.Parameter(torch.randn(1, num_cams, 1, d_model) * 1e-3)
        self.out_ln = nn.LayerNorm(d_model)
        self.gate = LayerScale(d_model, init_values=0.0, inplace=False) if use_gate else nn.Identity()

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        if geo.ndim != 4:
            raise ValueError(f"Expected VGGT geometry tokens with shape [B, 4, T, D], got {tuple(geo.shape)}")
        batch_size, num_cams, tokens_per_cam, _ = geo.shape
        if tokens_per_cam != self.tokens_per_cam:
            raise ValueError(f"Expected {self.tokens_per_cam} VGGT geometry tokens per camera, got {tokens_per_cam}")

        x = self.proj(self.input_ln(geo.float()))
        x = x + self.branch_embed + self.cam_embed[:, :num_cams]
        x = self.gate(self.out_ln(x))
        return x.reshape(batch_size, num_cams * tokens_per_cam, -1)


class FrozenVggtGeometryTeacher(nn.Module):
    """Frozen VGGT-Omega teacher used only for online geometry evaluation or cache generation."""

    def __init__(self, checkpoint_path: str, use_camera_token: bool = False, joint_forward: bool = True) -> None:
        super().__init__()
        from vggt_omega.models import VGGTOmega

        self.model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
        checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu")
        if isinstance(checkpoint, Mapping):
            checkpoint = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        checkpoint = {key.replace("module.", "", 1): value for key, value in checkpoint.items()}

        missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
        missing_backbone = [key for key in missing if key.startswith("aggregator.")]
        if missing_backbone:
            raise RuntimeError(f"VGGT-Omega checkpoint is missing aggregator keys: {missing_backbone[:8]}")
        unexpected = [
            key
            for key in unexpected
            if not key.startswith(("camera_head", "dense_head", "text_alignment_head"))
        ]
        if unexpected:
            logger.warning("Ignored unexpected VGGT-Omega checkpoint keys: %s", unexpected[:8])

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.use_camera_token = bool(use_camera_token)
        self.joint_forward = bool(joint_forward)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected teacher images with shape [B, 4, 3, H, W], got {tuple(images.shape)}")

        if self.joint_forward:
            tokens = self.model(images)["camera_and_register_tokens"]
        else:
            per_camera = [self.model(images[:, i : i + 1])["camera_and_register_tokens"] for i in range(images.shape[1])]
            tokens = torch.cat(per_camera, dim=1)
        return tokens if self.use_camera_token else tokens[:, :, 1:]


def preprocess_arrays_for_teacher(
    images_np: Sequence[np.ndarray],
    mode: str = "balanced",
    image_resolution: int = 512,
    patch_size: int = 16,
) -> torch.Tensor:
    """Use VGGT-Omega's official geometry resizing, replacing only Image.open with in-memory arrays."""

    from torchvision import transforms as TF
    from vggt_omega.utils.load_fn import (
        _balanced_target_shape,
        _crop_to_supported_aspect_ratio,
        _max_size_target_shape,
        _pad_images_to_common_size,
    )

    if mode not in ("balanced", "max_size"):
        raise ValueError("VGGT geometry teacher preprocess mode must be 'balanced' or 'max_size'")

    to_tensor = TF.ToTensor()
    images = []
    shapes = set()
    for array in images_np:
        image = _crop_to_supported_aspect_ratio(Image.fromarray(array).convert("RGB"))
        width, height = image.size
        aspect_ratio = height / max(width, 1)
        if mode == "balanced":
            target_h, target_w = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
        else:
            target_h, target_w = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)
        tensor = to_tensor(image.resize((target_w, target_h), Image.Resampling.BICUBIC))
        shapes.add((tensor.shape[1], tensor.shape[2]))
        images.append(tensor)

    if len(shapes) > 1:
        images = _pad_images_to_common_size(images, shapes)
    return torch.stack(images)


class VggtGeometryTokenProvider:
    """Load normal/shuffle tokens or generate noise tokens before projection."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        mode: str,
        shuffle_seed: int,
        expected_shape: Tuple[int, int, int],
        worker_offset: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.mode = str(mode)
        self.expected_shape = tuple(int(value) for value in expected_shape)
        self.index = _load_token_index(self.cache_dir)

        if self.mode not in ("normal", "shuffle", "noise", "drop"):
            raise ValueError(f"Unknown VGGT geometry mode: {mode}")

        seed = int(shuffle_seed) + int(worker_offset)
        if self.mode == "shuffle":
            self.all_tokens = sorted(self.index.keys()) if self.index else self._scan_tokens()
            if len(self.all_tokens) < 2:
                raise ValueError("VGGT geometry shuffle needs at least two cached tokens")
            self.rng = np.random.default_rng(seed)
            logger.info("VGGT geometry shuffle rng seed=%d", seed)

        if self.mode == "noise":
            stats = torch.load(self.cache_dir / STATS_FILENAME, map_location="cpu")
            self.mean = stats["mean"].float()
            self.std = stats["std"].float().clamp_min(1e-6)
            self.rng_t = torch.Generator().manual_seed(seed)

    def get(self, token: str) -> torch.Tensor:
        if self.mode == "noise":
            noise = torch.randn(self.expected_shape, generator=self.rng_t)
            return (self.mean.view(1, 1, -1) + self.std.view(1, 1, -1) * noise).to(CACHE_DTYPE)

        if self.mode == "shuffle":
            while True:
                partner = self.all_tokens[int(self.rng.integers(len(self.all_tokens)))]
                if partner != token:
                    token = partner
                    break

        tensor = _select_tensor(torch.load(self._resolve_token_path(token), map_location="cpu"))
        if tuple(tensor.shape) != self.expected_shape:
            raise ValueError(f"Bad VGGT geometry token shape for {token}: expected {self.expected_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != CACHE_DTYPE:
            raise ValueError(f"Bad VGGT geometry token dtype for {token}: expected {CACHE_DTYPE}, got {tensor.dtype}")
        return tensor

    def _resolve_token_path(self, token: str) -> Path:
        if self.index is not None and token in self.index:
            path = Path(self.index[token])
            return path if path.is_absolute() else self.cache_dir / path

        sharded = vggt_geometry_cache_file(self.cache_dir, token)
        if sharded.is_file():
            return sharded

        flat = self.cache_dir / f"{token}.pt"
        if flat.is_file():
            return flat

        raise FileNotFoundError(f"VGGT geometry token cache missing for token {token} in {self.cache_dir}")

    def _scan_tokens(self) -> List[str]:
        return sorted(path.stem for path in self.cache_dir.rglob("*.pt") if ".tmp." not in path.name)


def build_vggt_geometry_fingerprint(config: Mapping[str, Any]) -> str:
    """Compact fingerprint helper for VGGT geometry configs."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

