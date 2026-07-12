"""Adapter around the vendored LingBot-Vision ViT for DrivoR's ImgEncoder.

Exposes the same interface `ImgEncoder` and `_LoRA_qkv_timm` already rely on
for the timm/DINOv2 backbone: `forward_features(x, scene_tokens)`,
`num_features`, `patch_size`, and `blocks` (for the LoRA qkv surgery).
"""

from pathlib import Path

import torch
import torch.nn as nn

from lingbot_vision.build import build_backbone_from_cfg
from lingbot_vision.loader import extract_submodule, load_backbone_state, load_config, _unwrap_state

from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)

_VARIANT_CONFIGS = {
    "small": "lbot_vision_vits.yaml",
    "base": "lbot_vision_vitb.yaml",
    "large": "lbot_vision_vitl.yaml",
}


class LingBotBackbone(nn.Module):
    """Wraps `LingBotVisionTransformer` with scene-token injection.

    Scene tokens are placed at the very front of the sequence (before cls +
    storage + patch tokens), so RoPE's runtime prefix inference
    (`prefix = N - H*W`) treats them as position-free, matching the timm
    path's "scene tokens get no pos embed" semantics.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.num_features = model.embed_dim
        self.patch_size = model.patch_size

    @property
    def blocks(self):
        return self.model.blocks

    def forward_features(self, x: torch.Tensor, scene_tokens: torch.Tensor = None) -> torch.Tensor:
        model = self.model
        tokens, (H, W) = model.prepare_tokens_with_masks(x)
        if scene_tokens is not None:
            tokens = torch.cat([scene_tokens, tokens], dim=1)
        rope_sincos = model.rope_embed(H=H, W=W)
        for blk in model.blocks:
            tokens = blk(tokens, rope_sincos)
        return model.norm(tokens)


def build_lingbot_backbone(config) -> LingBotBackbone:
    """Build a trainable LingBot-Vision backbone from an `image_backbone` config.

    Deliberately does not use `lingbot_vision.load_pretrained_backbone`: that
    helper returns a frozen, bf16, eval-mode model. Instead this follows the
    step-by-step path (`load_config` -> `build_backbone_from_cfg` ->
    `load_backbone_state` + `load_state_dict`) so dtype stays fp32 and
    train/eval mode is left to `ImgEncoder`.
    """
    variant = config.get("variant", "small")
    if variant not in _VARIANT_CONFIGS:
        raise ValueError(f"Unknown lingbot variant: {variant!r}; expected one of {sorted(_VARIANT_CONFIGS)}")

    cfg = load_config(_VARIANT_CONFIGS[variant])
    if not config.get("rope_train_aug", False):
        cfg.student.lbot_vision.pos_embed_rope_rescale_coords = None

    model, _ = build_backbone_from_cfg(cfg)

    checkpoint_path = Path(config.model_weights) / "model.pt"
    ckpt = load_backbone_state(checkpoint_path)
    state_dict = extract_submodule(ckpt, "backbone.")
    if not state_dict:
        state_dict = _unwrap_state(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info(
        f"lingbot backbone loaded: variant={variant} weights={checkpoint_path} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )

    return LingBotBackbone(model)
