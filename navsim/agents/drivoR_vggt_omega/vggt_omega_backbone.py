import math
from contextlib import nullcontext
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import _LoRA_qkv_timm
from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask
from vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten


class SceneTokenAggregator(Aggregator):
    """VGGT-Omega aggregator with learnable scene tokens in the prefix."""

    def __init__(self, num_scene_tokens: int = 16, grad_checkpointing: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_scene_tokens = int(num_scene_tokens)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.scene_token_start = self.patch_token_start
        self.patch_token_start += self.num_scene_tokens

    def forward_full(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, N, 3, H, W], got {tuple(images.shape)}")
        if scene_tokens.ndim != 4:
            raise ValueError(
                f"Expected scene tokens with shape [B, N, S, D], got {tuple(scene_tokens.shape)}"
            )

        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")
        expected_scene_shape = (batch_size, num_frames, self.num_scene_tokens)
        if tuple(scene_tokens.shape[:3]) != expected_scene_shape:
            raise ValueError(
                f"Expected scene token prefix {expected_scene_shape}, got {tuple(scene_tokens.shape[:3])}"
            )

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)
        scene = scene_tokens.reshape(batch_size * num_frames, self.num_scene_tokens, -1)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, scene, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        grid = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=grid[0], W=grid[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        def run_block(block_tokens: torch.Tensor, block_idx: int):
            block_tokens, frame_tokens = self._run_frame_block(
                block_tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            block_tokens = self._run_inter_frame_attention_block(
                block_tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )
            return block_tokens, frame_tokens

        frame_tokens = None
        for block_idx in range(self.depth):
            if self.grad_checkpointing and torch.is_grad_enabled():
                tokens, frame_tokens = checkpoint(run_block, tokens, block_idx, use_reentrant=False)
            else:
                tokens, frame_tokens = run_block(tokens, block_idx)

        return torch.cat([frame_tokens, tokens], dim=-1)

    def forward(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        output = self.forward_full(images, scene_tokens)
        return output[:, :, self.scene_token_start : self.patch_token_start]


def apply_lora_to_blocks(blocks: nn.ModuleList, rank: int) -> list[nn.Module]:
    """Add DrivoR-style Q/V-only LoRA adapters to attention blocks."""

    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")

    lora_layers = []
    for block in blocks:
        qkv = block.attn.qkv
        dim = qkv.in_features
        a_q = nn.Linear(dim, rank, bias=False)
        b_q = nn.Linear(rank, dim, bias=False)
        a_v = nn.Linear(dim, rank, bias=False)
        b_v = nn.Linear(rank, dim, bias=False)
        for layer in (a_q, a_v):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        for layer in (b_q, b_v):
            nn.init.zeros_(layer.weight)
        block.attn.qkv = _LoRA_qkv_timm(
            qkv,
            a_q,
            b_q,
            a_v,
            b_v,
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
        )
        lora_layers.extend((a_q, b_q, a_v, b_v))
    return lora_layers


def load_aggregator_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load and normalize the aggregator portion of a VGGT-Omega checkpoint."""

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping):
        state = state.get("model", state.get("state_dict", state))
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported VGGT-Omega checkpoint type: {type(state).__name__}")
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    aggregator_state = {
        key[len("aggregator.") :]: value for key, value in state.items() if key.startswith("aggregator.")
    }
    if not aggregator_state:
        raise RuntimeError("VGGT-Omega checkpoint does not contain any aggregator.* keys")
    return aggregator_state


class VggtOmegaImgEncoder(nn.Module):
    """Frozen VGGT-Omega backbone with optional Q/V-only LoRA adapters."""

    VGGT_EMBED_DIM = 1024
    READOUT_DIM = 2048
    LORA_TARGETS = ("frame", "inter_frame")

    def __init__(self, config) -> None:
        super().__init__()
        self.num_features = self.VGGT_EMBED_DIM
        self.aggregator = SceneTokenAggregator(
            num_scene_tokens=config["num_scene_tokens"],
            grad_checkpointing=config.get("grad_checkpointing", True),
        )
        self.aggregator.load_state_dict(load_aggregator_state_dict(config["checkpoint_path"]), strict=True)
        self.aggregator.requires_grad_(False)

        self.use_lora = bool(config.get("use_lora", False))
        if self.use_lora:
            targets = {
                "frame": self.aggregator.frame_blocks,
                "inter_frame": self.aggregator.inter_frame_blocks,
            }
            requested_targets = list(config.get("lora_targets", self.LORA_TARGETS))
            if len(requested_targets) != len(set(requested_targets)):
                raise ValueError(f"Duplicate LoRA targets are not allowed: {requested_targets}")
            for name in requested_targets:
                if name not in targets:
                    raise ValueError(f"Unsupported LoRA target {name!r}; allowed targets: {sorted(targets)}")
                apply_lora_to_blocks(targets[name], rank=int(config.get("lora_rank", 32)))

        self.neck = nn.Linear(self.READOUT_DIM, config["tf_d_model"])
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = bool(config.get("use_grid_mask", False))

    def train(self, mode: bool = True):
        super().train(mode)
        self.aggregator.eval()
        return self

    def forward(self, img: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = img.shape[0]
        if self.use_grid_mask and self.training:
            img = self.grid_mask(img.flatten(0, 1)).view_as(img)

        if img.is_cuda:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            amp_context = nullcontext()
        with amp_context:
            geometry_tokens = self.aggregator(img, scene_tokens)

        tokens = self.neck(geometry_tokens.float())
        return tokens.reshape(batch_size, -1, tokens.shape[-1])
