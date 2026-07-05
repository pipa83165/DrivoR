#!/usr/bin/env python3
"""Lightweight VGGT geometry acceptance checks.

This does not load VGGT-Omega weights and does not run training. It checks the
small geometry-token behaviors that should be true before expensive cache generation.
"""

import tempfile
from pathlib import Path

import torch

from navsim.agents.drivoR.vggt_geometry import VggtGeometryTokenProvider, VggtGeometryProjector, vggt_geometry_cache_file


def check_projector_zero() -> None:
    proj = VggtGeometryProjector(vggt_dim=8, d_model=4, num_cams=4, tokens_per_cam=2)
    out = proj(torch.randn(2, 4, 2, 8) * 30)
    assert out.abs().max().item() == 0.0, "VggtGeometryProjector should output exact zeros at init"
    print("OK: VggtGeometryProjector cold-start output is exactly zero")


def check_drop_memory() -> None:
    from navsim.agents.drivoR.drivor_model import DrivoRModel

    class Dummy:
        pass

    dummy = Dummy()
    dummy.vggt_geometry_mode = "drop"
    dummy.vggt_geometry_source = "cache"
    dummy.training = False
    scene = torch.randn(1, 64, 4)
    out = DrivoRModel._extend_memory_with_vggt_geometry(dummy, scene, {})
    assert out.shape[1] == 64, "Drop-mode eval should keep memory length at 64"

    dummy.training = True
    dummy.vggt_geometry_projector = VggtGeometryProjector(vggt_dim=8, d_model=4, num_cams=4, tokens_per_cam=2)
    out = DrivoRModel._extend_memory_with_vggt_geometry(dummy, scene, {"vggt_geometry_tokens": torch.randn(1, 4, 2, 8).half()})
    assert out.shape[1] == 72, "Drop-mode training should still append geometry tokens"
    print("OK: drop mode physically removes geometry tokens only in eval")


def check_provider_modes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        tokens = {"aa00000000000000": torch.ones(4, 2, 8).half(), "bb00000000000000": torch.zeros(4, 2, 8).half()}
        index = {}
        for token, tensor in tokens.items():
            path = vggt_geometry_cache_file(cache_dir, token)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path)
            index[token] = str(path.relative_to(cache_dir))
        (cache_dir / "token_index.json").write_text(__import__("json").dumps(index))
        torch.save({"mean": torch.zeros(8), "std": torch.ones(8)}, cache_dir / "noise_stats.pt")

        normal = VggtGeometryTokenProvider(cache_dir, "normal", 7, (4, 2, 8))
        assert normal.get("aa00000000000000").shape == (4, 2, 8)

        shuffle = VggtGeometryTokenProvider(cache_dir, "shuffle", 7, (4, 2, 8))
        shuffled = shuffle.get("aa00000000000000")
        assert shuffled.shape == (4, 2, 8)
        assert not torch.equal(shuffled, tokens["aa00000000000000"]), "Shuffle partner should differ from self"

        noise = VggtGeometryTokenProvider(cache_dir, "noise", 7, (4, 2, 8))
        assert noise.get("aa00000000000000").shape == (4, 2, 8)
    print("OK: VggtGeometryTokenProvider normal/shuffle/noise modes work")


def main() -> None:
    check_projector_zero()
    check_drop_memory()
    check_provider_modes()
    print("PASS: lightweight VGGT geometry acceptance checks passed")


if __name__ == "__main__":
    main()
