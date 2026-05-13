from __future__ import annotations

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
            "output_dir": str(tmp_path / "outputs"),
        },
        "garment": {
            "target_width": 32,
            "target_height": 32,
            "min_width": 1,
            "min_height": 1,
            "max_file_mb": 20,
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
