#!/usr/bin/env python3
"""Focused acceptance checks for the DrivoR VGGT-Omega backbone."""

import argparse
import gc
from pathlib import Path

import torch

from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import _LoRA_qkv_timm
from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR_vggt_omega.vggt_omega_backbone import (
    SceneTokenAggregator,
    VggtOmegaImgEncoder,
    apply_lora_to_blocks,
    load_aggregator_state_dict,
)
from vggt_omega.models import VGGTOmega


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/vggt_omega_1b_512.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=688)
    parser.add_argument(
        "--agent-config",
        default="navsim/planning/script/config/common/agent/drivoR_vggt_omega.yaml",
    )
    parser.add_argument(
        "--check",
        choices=("all", "lora", "encoder", "memory", "official"),
        default="all",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def check_lora(checkpoint_path: str, device: torch.device) -> None:
    aggregator = SceneTokenAggregator(grad_checkpointing=False)
    aggregator.load_state_dict(load_aggregator_state_dict(checkpoint_path), strict=True)
    aggregator.requires_grad_(False).to(device).eval()

    blocks = list(aggregator.frame_blocks) + list(aggregator.inter_frame_blocks)
    inputs = []
    outputs_before = []
    for block in blocks:
        sample = torch.randn(1, 2, block.attn.qkv.in_features, device=device)
        inputs.append(sample)
        with torch.no_grad():
            outputs_before.append(block.attn.qkv(sample).clone())

    apply_lora_to_blocks(aggregator.frame_blocks, rank=32)
    apply_lora_to_blocks(aggregator.inter_frame_blocks, rank=32)

    wrappers = [module for module in aggregator.modules() if isinstance(module, _LoRA_qkv_timm)]
    if len(wrappers) != 48:
        raise AssertionError(f"Expected 48 Q/V LoRA wrappers, got {len(wrappers)}")
    if any(isinstance(module, _LoRA_qkv_timm) for module in aggregator.patch_embed.modules()):
        raise AssertionError("DINOv3 trunk must not contain LoRA wrappers")

    loss = torch.zeros((), device=device)
    for block, sample, before in zip(blocks, inputs, outputs_before):
        after = block.attn.qkv(sample)
        dim = block.attn.qkv.dim
        if not torch.equal(before, after):
            raise AssertionError("Zero-initialized LoRA changed qkv output")
        if not torch.equal(before[:, :, dim : 2 * dim], after[:, :, dim : 2 * dim]):
            raise AssertionError("Q/V-only LoRA changed the K slice")
        loss = loss + after.sum()
    loss.backward()

    lora_parameters = [
        parameter
        for name, parameter in aggregator.named_parameters()
        if any(part in name for part in ("linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v"))
    ]
    if len(lora_parameters) != 48 * 4:
        raise AssertionError(f"Expected 192 LoRA parameter tensors, got {len(lora_parameters)}")
    if any(parameter.grad is None for parameter in lora_parameters):
        raise AssertionError("At least one LoRA parameter did not receive a gradient")
    print("LoRA checks passed: 48 Q/V-only targets, unchanged K slices, no trunk adapters")

    del aggregator
    clear_device(device)


def check_encoder(checkpoint_path: str, device: torch.device, height: int, width: int) -> None:
    config = {
        "checkpoint_path": checkpoint_path,
        "num_scene_tokens": 16,
        "tf_d_model": 256,
        "grad_checkpointing": True,
        "use_lora": False,
        "use_grid_mask": False,
    }
    encoder = VggtOmegaImgEncoder(config).to(device).train()
    images = torch.rand(1, 4, 3, height, width, device=device)
    scene_tokens = torch.randn(1, 4, 16, 1024, device=device, requires_grad=True) * 1e-6
    scene_tokens.retain_grad()
    output = encoder(images, scene_tokens)
    if output.shape != (1, 64, 256):
        raise AssertionError(f"Expected encoder output [1, 64, 256], got {tuple(output.shape)}")
    output.sum().backward()
    if scene_tokens.grad is None:
        raise AssertionError("Scene tokens did not receive gradients through the frozen backbone")
    if encoder.neck.weight.grad is None:
        raise AssertionError("Readout neck did not receive gradients")
    if any(parameter.grad is not None for parameter in encoder.aggregator.parameters()):
        raise AssertionError("Frozen aggregator parameters unexpectedly received gradients")
    if encoder.aggregator.training:
        raise AssertionError("Aggregator must remain in eval mode while the encoder trains")
    synchronize(device)
    print("Encoder checks passed: [1, 64, 256] output and frozen-backbone gradient path")

    del encoder, images, scene_tokens, output
    clear_device(device)


def check_official_parity(checkpoint_path: str, device: torch.device, height: int, width: int) -> None:
    state = load_aggregator_state_dict(checkpoint_path)
    custom = SceneTokenAggregator(num_scene_tokens=0, grad_checkpointing=False)
    custom.load_state_dict(state, strict=True)
    official = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
    official.aggregator.load_state_dict(state, strict=True)
    custom.to(device).eval()
    official.to(device).eval()

    images = torch.rand(1, 4, 3, height, width, device=device)
    empty_scene = torch.empty(1, 4, 0, 1024, device=device)
    with torch.no_grad():
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                official_outputs, patch_token_start = official.aggregator(images)
                expected = official_outputs[-1][:, :, :patch_token_start]
                actual = custom.forward_full(images, empty_scene)[:, :, :17]
        else:
            official_outputs, patch_token_start = official.aggregator(images)
            expected = official_outputs[-1][:, :, :patch_token_start]
            actual = custom.forward_full(images, empty_scene)[:, :, :17]
    cosine = torch.nn.functional.cosine_similarity(actual.float(), expected.float(), dim=-1).min().item()
    if cosine <= 0.999:
        raise AssertionError(f"Official parity cosine must exceed 0.999, got {cosine:.6f}")
    print(f"Official S=0 parity passed: minimum token cosine={cosine:.6f}")


def check_decoder_memory(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    height: int,
    width: int,
) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path).config
    config.image_backbone.checkpoint_path = checkpoint_path
    model = DrivoRModel(config).to(device).eval()
    captured_memory_lengths = []

    def capture_memory(_module, inputs) -> None:
        captured_memory_lengths.append(inputs[1].shape[1])

    hook = model.trajectory_decoder.register_forward_pre_hook(capture_memory)
    features = {
        "image": torch.rand(1, 4, 3, height, width, device=device),
        "ego_status": torch.zeros(1, 1, 11, device=device),
    }
    with torch.no_grad():
        model(features)
    hook.remove()
    if captured_memory_lengths != [64]:
        raise AssertionError(f"Expected decoder memory length [64], got {captured_memory_lengths}")
    print("Decoder memory check passed: VGGT-Omega backbone supplies exactly 64 tokens")

    del model, features
    clear_device(device)


def main() -> None:
    args = parse_args()
    checkpoint_path = str(Path(args.checkpoint).expanduser())
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA acceptance checks requested, but CUDA is unavailable")
    if args.height % 16 or args.width % 16:
        raise ValueError("Acceptance image dimensions must be divisible by patch size 16")

    if args.check in ("all", "lora"):
        check_lora(checkpoint_path, device)
    if args.check in ("all", "encoder"):
        check_encoder(checkpoint_path, device, args.height, args.width)
    if args.check in ("all", "memory"):
        check_decoder_memory(checkpoint_path, args.agent_config, device, args.height, args.width)
    if args.check in ("all", "official"):
        check_official_parity(checkpoint_path, device, args.height, args.width)


if __name__ == "__main__":
    main()
