from __future__ import annotations

import io
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


def _make_fal_cfg(model: str = "fal-ai/cat-vton") -> dict:
    return {
        "vton": {"adapter": "fal_api", "num_inference_steps": 1, "guidance_scale": 2.5},
        "fal_api": {"model": model},
    }


def _fake_result_image() -> dict:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (0, 128, 0)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@patch.dict("os.environ", {"FAL_KEY": "test-key"})
@patch("fal_client.upload", return_value="https://fal.ai/fake-upload-url")
@patch("fal_client.run")
def test_fal_cat_vton_calls_correct_endpoint(mock_run, mock_upload, tiny_rgb, tiny_mask):
    import requests_mock as req_mock_lib

    result_png = _fake_result_image()
    mock_run.return_value = {"image": {"url": "https://fal.ai/result.png"}}

    with req_mock_lib.Mocker() as m:
        m.get("https://fal.ai/result.png", content=result_png)

        from pipeline.vton.fal_api import FalAPIAdapter
        adapter = FalAPIAdapter(_make_fal_cfg("fal-ai/cat-vton"))
        result = adapter.generate(tiny_rgb, tiny_mask, tiny_rgb, tiny_mask, {}, "upper_body")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == "fal-ai/cat-vton"
    assert "human_image_url" in call_args[1]["arguments"]
    assert "garment_image_url" in call_args[1]["arguments"]
    assert isinstance(result, Image.Image)


@patch.dict("os.environ", {"FAL_KEY": "test-key"})
@patch("fal_client.upload", return_value="https://fal.ai/fake-upload-url")
@patch("fal_client.run")
def test_fal_idm_vton_passes_category(mock_run, mock_upload, tiny_rgb, tiny_mask):
    import requests_mock as req_mock_lib

    result_png = _fake_result_image()
    mock_run.return_value = {"image": {"url": "https://fal.ai/result.png"}}

    with req_mock_lib.Mocker() as m:
        m.get("https://fal.ai/result.png", content=result_png)

        from pipeline.vton.fal_api import FalAPIAdapter
        adapter = FalAPIAdapter(_make_fal_cfg("fal-ai/idm-vton"))
        adapter.generate(tiny_rgb, tiny_mask, tiny_rgb, tiny_mask, {}, "lower_body")

    args = mock_run.call_args[1]["arguments"]
    assert args["category"] == "lower"


@patch.dict("os.environ", {"FAL_KEY": "test-key"})
@patch("fal_client.upload", return_value="https://fal.ai/fake-upload-url")
@patch("fal_client.run")
def test_fal_handles_images_list_response(mock_run, mock_upload, tiny_rgb, tiny_mask):
    import requests_mock as req_mock_lib

    result_png = _fake_result_image()
    mock_run.return_value = {"images": [{"url": "https://fal.ai/result.png"}]}

    with req_mock_lib.Mocker() as m:
        m.get("https://fal.ai/result.png", content=result_png)

        from pipeline.vton.fal_api import FalAPIAdapter
        adapter = FalAPIAdapter(_make_fal_cfg())
        result = adapter.generate(tiny_rgb, tiny_mask, tiny_rgb, tiny_mask, {}, "upper_body")

    assert isinstance(result, Image.Image)


@patch.dict("os.environ", {}, clear=True)
def test_fal_raises_without_api_key():
    from pipeline.vton.fal_api import FalAPIAdapter
    with pytest.raises(EnvironmentError, match="FAL_KEY"):
        FalAPIAdapter(_make_fal_cfg())


def test_fal_raises_on_unsupported_model():
    import os
    with patch.dict(os.environ, {"FAL_KEY": "test"}):
        from pipeline.vton.fal_api import FalAPIAdapter
        with pytest.raises(ValueError, match="Unsupported fal.ai model"):
            FalAPIAdapter({**_make_fal_cfg(), "fal_api": {"model": "fal-ai/unknown-model"}})
