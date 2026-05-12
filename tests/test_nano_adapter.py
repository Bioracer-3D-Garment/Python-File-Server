from __future__ import annotations

import pytest
from PIL import Image

from pipeline.vton.nano import NanoAdapter


@pytest.fixture()
def nano(sample_config):
    return NanoAdapter(sample_config)


def test_nano_fulfils_contract(nano, tiny_rgb, tiny_rgba, tiny_mask):
    result = nano.generate(
        garment=tiny_rgba,
        garment_mask=tiny_mask,
        person=tiny_rgb,
        agnostic_mask=tiny_mask,
        pose_data={},
        category="upper_body",
    )
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == tiny_rgb.size


def test_nano_handles_all_categories(nano, tiny_rgb, tiny_rgba, tiny_mask):
    for cat in ("upper_body", "lower_body", "dresses"):
        result = nano.generate(tiny_rgba, tiny_mask, tiny_rgb, tiny_mask, {}, cat)
        assert result.mode == "RGB"
        assert result.size == tiny_rgb.size


def test_nano_empty_agnostic_mask_returns_person(nano, tiny_rgb, tiny_rgba):
    blank_mask = Image.new("L", tiny_rgb.size, 0)
    result = nano.generate(tiny_rgba, blank_mask, tiny_rgb, blank_mask, {}, "upper_body")
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == tiny_rgb.size


def test_nano_via_build_adapter(sample_config):
    from pipeline.vton.base import build_adapter
    sample_config["vton"]["adapter"] = "nano"
    adapter = build_adapter(sample_config)
    assert isinstance(adapter, NanoAdapter)
