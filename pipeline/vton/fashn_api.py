from __future__ import annotations

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

    def generate(
        self,
        garment: Image.Image,
        person: Image.Image,
        category: str,
    ) -> Image.Image:
        files = {
            "model_image": ("person.png", pil_to_bytes(person), "image/png"),
            "garment_image": ("garment.png", pil_to_bytes(garment), "image/png"),
        }
        data = {"category": self._map_category(category)}

        resp = requests.post(
            f"{self._base_url}/run",
            headers=self._headers,
            files=files,
            data=data,
            timeout=self._timeout,
        )
        resp.raise_for_status()
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
