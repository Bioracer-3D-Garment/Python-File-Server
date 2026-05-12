from __future__ import annotations

from typing import Any

from PIL import Image

from pipeline.vton.base import VTONAdapter


class OOTDiffusionAdapter(VTONAdapter):
    """
    Local adapter for OOTDiffusion.
    https://github.com/levihsu/OOTDiffusion

    Setup:
        git clone https://github.com/levihsu/OOTDiffusion
        pip install -r OOTDiffusion/requirements.txt
        # Download weights from HuggingFace: levihsu/OOTDiffusion

    Set OOTD_REPO_PATH to the cloned repo root.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._pipe = None  # lazy-loaded on first call

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import sys
        import os

        repo_path = os.environ.get("OOTD_REPO_PATH", "OOTDiffusion")
        sys.path.insert(0, repo_path)

        from run_ootd import OOTDiffusion  # type: ignore

        self._pipe = OOTDiffusion(repo_path)

    def generate(
        self,
        garment: Image.Image,
        garment_mask: Image.Image,
        person: Image.Image,
        agnostic_mask: Image.Image,
        pose_data: dict[str, Any],
        category: str,
    ) -> Image.Image:
        self._load()

        vton_cfg = self._cfg.get("vton", {})
        steps = vton_cfg.get("num_inference_steps", 20)
        guidance = vton_cfg.get("guidance_scale", 2.0)

        category_map = {"upper_body": 0, "lower_body": 1, "dresses": 2}
        cat_idx = category_map.get(category, 0)

        images = self._pipe(
            model_type="hd",
            category=cat_idx,
            image_garm=garment,
            image_vton=person,
            mask=agnostic_mask,
            image_ori=person,
            num_samples=1,
            num_steps=steps,
            guidance_scale=guidance,
        )
        return images[0]
