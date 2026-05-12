from __future__ import annotations

import io
import os
from typing import Any

import requests
from PIL import Image

from pipeline.vton.base import VTONAdapter
from pipeline.utils.image import pil_to_bytes

# fal.ai model endpoint identifiers
FAL_IDM_VTON = "fal-ai/idm-vton"
FAL_CAT_VTON = "fal-ai/cat-vton"

SUPPORTED_MODELS = (FAL_IDM_VTON, FAL_CAT_VTON)


class FalAPIAdapter(VTONAdapter):
    """
    Virtual try-on via fal.ai serverless endpoints.
    Supports IDM-VTON and CatVTON — select via config/pipeline.yaml → fal_api.model.

    Requires:
        pip install fal-client
        FAL_KEY environment variable (from fal.ai dashboard)

    CatVTON note: lighter model, ignores agnostic_mask/pose_data (handled server-side).
    IDM-VTON note: more faithful pattern reproduction; passes category hint.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        api_cfg = cfg.get("fal_api", {})
        self._model = api_cfg.get("model", FAL_CAT_VTON)
        if self._model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported fal.ai model: {self._model!r}. Choose from: {SUPPORTED_MODELS}")

        self._num_steps = cfg.get("vton", {}).get("num_inference_steps", 30)
        self._guidance = cfg.get("vton", {}).get("guidance_scale", 2.5)

        api_key = os.environ.get("FAL_KEY", "")
        if not api_key:
            raise EnvironmentError("FAL_KEY environment variable is not set.")

        try:
            import fal_client  # type: ignore
            self._fal = fal_client
        except ImportError as exc:
            raise ImportError("fal-client is required: pip install fal-client") from exc

    def generate(
        self,
        garment: Image.Image,
        garment_mask: Image.Image,
        person: Image.Image,
        agnostic_mask: Image.Image,
        pose_data: dict[str, Any],
        category: str,
    ) -> Image.Image:
        garment_url = self._upload(garment)
        person_url = self._upload(person)

        if self._model == FAL_CAT_VTON:
            arguments = self._cat_vton_args(person_url, garment_url)
        else:
            arguments = self._idm_vton_args(person_url, garment_url, category)

        result = self._fal.run(self._model, arguments=arguments)
        return self._download(result)

    def _upload(self, img: Image.Image) -> str:
        data = pil_to_bytes(img, fmt="PNG")
        return self._fal.upload(data, content_type="image/png")

    def _cat_vton_args(self, person_url: str, garment_url: str) -> dict:
        return {
            "human_image_url": person_url,
            "garment_image_url": garment_url,
            "num_inference_steps": self._num_steps,
            "guidance_scale": self._guidance,
        }

    def _idm_vton_args(self, person_url: str, garment_url: str, category: str) -> dict:
        category_map = {
            "upper_body": "upper",
            "lower_body": "lower",
            "dresses": "dresses",
        }
        return {
            "human_image_url": person_url,
            "garment_image_url": garment_url,
            "garment_description": "",
            "category": category_map.get(category, "upper"),
            "num_inference_steps": self._num_steps,
            "guidance_scale": self._guidance,
        }

    def _download(self, result: dict) -> Image.Image:
        # fal.ai returns {"image": {"url": "..."}} or {"images": [{"url": "..."}]}
        if "image" in result:
            url = result["image"]["url"]
        elif "images" in result:
            url = result["images"][0]["url"]
        else:
            raise RuntimeError(f"Unexpected fal.ai response shape: {list(result.keys())}")

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def __repr__(self) -> str:
        return f"FalAPIAdapter(model={self._model!r})"
