from __future__ import annotations

import json
from pathlib import Path
import pytest
from PIL import Image


@pytest.fixture()
def tiny_rgb() -> Image.Image:
    """32×32 solid red RGB image."""
    return Image.new("RGB", (32, 32), (200, 50, 50))


@pytest.fixture()
def tiny_rgba() -> Image.Image:
    """32×32 RGBA image with full alpha."""
    return Image.new("RGBA", (32, 32), (200, 50, 50, 255))


@pytest.fixture()
def tiny_mask() -> Image.Image:
    """32×32 white binary mask."""
    return Image.new("L", (32, 32), 255)


@pytest.fixture()
def sample_config(tmp_path) -> dict:
    return {
        "pipeline": {
            "workers": 1,
            "input_dir": str(tmp_path / "inputs"),
            "poses_dir": str(tmp_path / "poses"),
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "outputs"),
        },
        "garment": {
            "target_width": 32,
            "target_height": 32,
            "background_threshold": 0.90,
        },
        "vton": {
            "adapter": "fashn_api",
            "num_inference_steps": 1,
            "guidance_scale": 1.0,
        },
        "postprocess": {
            "quality_threshold": 0.0,
            "output_format": "PNG",
        },
        "fashn_api": {
            "base_url": "https://api.fashn.ai/v1",
            "timeout": 10,
        },
    }


@pytest.fixture()
def pose_cache(tmp_path, tiny_rgb, tiny_mask) -> tuple[Path, str]:
    """Create a minimal pose cache entry and return (cache_dir, pose_id)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pose_id = "pose_test"
    tiny_rgb.save(cache_dir / f"{pose_id}_person.png")
    tiny_mask.save(cache_dir / f"{pose_id}_agnostic.png")
    tiny_mask.save(cache_dir / f"{pose_id}_parse.png")
    (cache_dir / f"{pose_id}_keypoints.json").write_text(json.dumps({}))
    return cache_dir, pose_id
