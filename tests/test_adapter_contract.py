from __future__ import annotations

import pytest
from PIL import Image

from pipeline.vton.base import VTONAdapter, build_adapter


class _MockAdapter(VTONAdapter):
    """Returns a solid green image — used to verify the adapter contract."""

    def generate(self, garment, garment_mask, person, agnostic_mask, pose_data, category):
        assert isinstance(garment, Image.Image)
        assert isinstance(garment_mask, Image.Image)
        assert isinstance(person, Image.Image)
        assert isinstance(agnostic_mask, Image.Image)
        assert isinstance(pose_data, dict)
        assert category in ("upper_body", "lower_body", "dresses")
        return Image.new("RGB", person.size, (0, 200, 0))


def test_mock_adapter_fulfils_contract(tiny_rgb, tiny_mask):
    adapter = _MockAdapter()
    result = adapter.generate(
        garment=tiny_rgb,
        garment_mask=tiny_mask,
        person=tiny_rgb,
        agnostic_mask=tiny_mask,
        pose_data={},
        category="upper_body",
    )
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == tiny_rgb.size


def test_build_adapter_raises_on_unknown_name():
    cfg = {"vton": {"adapter": "does_not_exist"}}
    with pytest.raises(ValueError, match="Unknown VTON adapter"):
        build_adapter(cfg)


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        VTONAdapter()  # type: ignore
