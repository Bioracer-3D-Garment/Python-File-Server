from __future__ import annotations

import pytest
from PIL import Image

from pipeline.preprocess.garment import validate_garment_image


@pytest.fixture()
def val_cfg():
    return {"garment": {"min_width": 100, "min_height": 100, "max_file_mb": 20}}


def test_valid_image_passes(tmp_path, val_cfg):
    img = Image.new("RGB", (200, 300), (100, 100, 100))
    p = tmp_path / "jersey.png"
    img.save(p)
    validate_garment_image(p, val_cfg)  # must not raise


def test_valid_jpeg_passes(tmp_path, val_cfg):
    img = Image.new("RGB", (200, 300))
    p = tmp_path / "jersey.jpg"
    img.save(p)
    validate_garment_image(p, val_cfg)


def test_unsupported_format_raises(tmp_path, val_cfg):
    img = Image.new("RGB", (200, 300))
    p = tmp_path / "jersey.tiff"
    img.save(p)
    with pytest.raises(ValueError, match="Unsupported format"):
        validate_garment_image(p, val_cfg)


def test_file_too_large_raises(tmp_path):
    cfg = {"garment": {"min_width": 1, "min_height": 1, "max_file_mb": 0.00001}}
    img = Image.new("RGB", (200, 300))
    p = tmp_path / "jersey.png"
    img.save(p)
    with pytest.raises(ValueError, match="too large"):
        validate_garment_image(p, cfg)


def test_image_too_small_raises(tmp_path, val_cfg):
    img = Image.new("RGB", (10, 10))
    p = tmp_path / "tiny.png"
    img.save(p)
    with pytest.raises(ValueError, match="too small"):
        validate_garment_image(p, val_cfg)


def test_corrupt_file_raises(tmp_path, val_cfg):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"this is not an image just garbage")
    with pytest.raises(ValueError, match="Cannot open image"):
        validate_garment_image(p, val_cfg)
