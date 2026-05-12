from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pipeline.utils.image import binary_mask_from_alpha, pad_to_aspect, pil_to_bytes


def test_pil_to_bytes_roundtrip(tiny_rgb):
    data = pil_to_bytes(tiny_rgb)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_pad_to_aspect_preserves_content(tiny_rgb):
    result = pad_to_aspect(tiny_rgb, 64, 64)
    assert result.size == (64, 64)
    assert result.mode == "RGB"


def test_pad_to_aspect_does_not_upscale_beyond_target():
    img = Image.new("RGB", (10, 10), (0, 255, 0))
    result = pad_to_aspect(img, 100, 100)
    assert result.size == (100, 100)


def test_binary_mask_from_alpha_full_alpha(tiny_rgba):
    mask = binary_mask_from_alpha(tiny_rgba)
    arr = np.array(mask)
    assert arr.min() == 255, "Full alpha should produce all-white mask"


def test_binary_mask_from_alpha_zero_alpha():
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    mask = binary_mask_from_alpha(img)
    arr = np.array(mask)
    assert arr.max() == 0, "Zero alpha should produce all-black mask"


def test_binary_mask_from_alpha_partial():
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 100))
    mask = binary_mask_from_alpha(img, threshold=127)
    arr = np.array(mask)
    assert arr.max() == 0, "Alpha=100 below threshold=127 should be black"
