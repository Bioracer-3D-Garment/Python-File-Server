from __future__ import annotations

import base64
import io
import os
import time

import requests
from PIL import Image

from pipeline.utils.image import pil_to_bytes


class FashnAPIAdapter:
    """
    Virtual try-on via the Fashn.ai REST API.
    Docs: https://fashn.ai/docs
    Set FASHN_API_KEY in environment.
    """

    def __init__(self, cfg: dict) -> None:
        api_cfg = cfg.get("fashn_api", {})
        self._base_url = api_cfg.get("base_url", "https://api.fashn.ai/v1").rstrip("/")
        self._timeout = api_cfg.get("timeout", 120)
        self._api_key = os.environ.get("FASHN_API_KEY", "")
        if not self._api_key:
            raise EnvironmentError("FASHN_API_KEY environment variable is not set.")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    @staticmethod
    def _to_data_uri(img: Image.Image) -> str:
        b64 = base64.b64encode(pil_to_bytes(img)).decode()
        return f"data:image/png;base64,{b64}"

    def generate(
        self,
        garment: Image.Image,
        person: Image.Image,
        category: str = "",  # unused by tryon-max; kept for interface compatibility
        prompt: str | None = None,
    ) -> Image.Image:
        # tryon-v1.6 (previous model)
        # payload = {
        #     "model_name": "tryon-v1.6",
        #     "inputs": {
        #         "model_image": self._to_data_uri(person),
        #         "garment_image": self._to_data_uri(garment),
        #         "category": self._map_category(category),
        #     },
        # }

        _prompt = prompt or (
            "Fit the garment onto the model exactly as shown in the product image. "
            "Preserve all text, logos, graphics, colors, patterns, and fabric details "
            "on the garment with pixel-accurate fidelity — do not alter, distort, "
            "remove, or reinterpret any design elements. "
            "The garment contains the text 'Bioracer' and 'Discontour .0.4' — "
            "reproduce these exactly as printed, preserving font style, size, color, and position. "
            "Do not modify the model's face, skin tone, hair, pose, or body in any way. "
            "The garment should appear naturally worn with realistic draping, fit, and lighting "
            "consistent with the model image."
        )

        # --- DEVELOPMENT (active) ---
        inputs: dict = {
            "model_image": self._to_data_uri(person),
            "product_image": self._to_data_uri(garment),
            "prompt": _prompt,
            "resolution": "4k",
            "generation_mode": "quality",
            "num_images": 1,
            "output_format": "png",
        }

        # --- PRODUCTION (max quality) ---
        # inputs: dict = {
        #     "model_image": self._to_data_uri(person),
        #     "product_image": self._to_data_uri(garment),
        #     "prompt": _prompt,
        #     "resolution": "4k",
        #     "generation_mode": "quality",
        #     "num_images": 1,
        #     "output_format": "png",
        # }

        payload = {
            "model_name": "tryon-max",
            "inputs": inputs,
        }

        resp = requests.post(
            f"{self._base_url}/run",
            headers={**self._headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self._timeout,
        )
        if not resp.ok:
            raise RuntimeError(f"Fashn.ai {resp.status_code}: {resp.text}")
        prediction_id = resp.json()["id"]

        return self._poll(prediction_id)

    def _poll(self, prediction_id: str) -> Image.Image:
        url = f"{self._base_url}/status/{prediction_id}"
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            resp = requests.get(url, headers=self._headers, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            status = body.get("status")
            if status == "completed":
                image_url = body["output"][0]
                img_resp = requests.get(image_url, timeout=60)
                img_resp.raise_for_status()
                return Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"Fashn.ai prediction {prediction_id} ended with status: {status}")
            time.sleep(3)
        raise TimeoutError(f"Fashn.ai prediction {prediction_id} did not complete within {self._timeout}s")

    @staticmethod
    def _map_category(category: str) -> str:
        mapping = {
            "upper_body": "tops",
            "lower_body": "bottoms",
            "dresses": "one-pieces",
        }
        return mapping.get(category, "tops")

    def __repr__(self) -> str:
        return self.__class__.__name__
