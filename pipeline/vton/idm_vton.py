from __future__ import annotations

from typing import Any

from PIL import Image

from pipeline.vton.base import VTONAdapter


class IDMVTONAdapter(VTONAdapter):
    """
    Local adapter for IDM-VTON.
    https://github.com/yisol/IDM-VTON

    Setup:
        git clone https://github.com/yisol/IDM-VTON
        pip install -r IDM-VTON/requirements.txt
        # Download weights to weights/idm_vton/ from HuggingFace: yisol/IDM-VTON

    Set IDMVTON_REPO_PATH to the cloned repo root if not adjacent to this project.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._pipe = None  # lazy-loaded on first call

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import sys
        import os
        import torch

        repo_path = os.environ.get("IDMVTON_REPO_PATH", "IDM-VTON")
        sys.path.insert(0, repo_path)

        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline  # type: ignore
        from src.unet_hacked_garmnet import UNet2DConditionModel as GarmentUNet  # type: ignore
        from src.unet_hacked_tryon import UNet2DConditionModel as TryonUNet  # type: ignore
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from diffusers import AutoencoderKL

        base = os.path.join(repo_path, "weights", "idm_vton")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        unet = TryonUNet.from_pretrained(base, subfolder="unet", torch_dtype=torch.float16).to(device)
        garment_unet = GarmentUNet.from_pretrained(base, subfolder="unet_encoder", torch_dtype=torch.float16).to(device)
        vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16).to(device)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(base, subfolder="image_encoder").to(device)
        processor = CLIPImageProcessor()

        self._pipe = TryonPipeline.from_pretrained(
            base,
            unet=unet,
            vae=vae,
            garment_feature_extractor=processor,
            image_encoder=image_encoder,
            garment_unet=garment_unet,
            torch_dtype=torch.float16,
        ).to(device)

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

        import torch
        vton_cfg = self._cfg.get("vton", {})
        steps = vton_cfg.get("num_inference_steps", 30)
        guidance = vton_cfg.get("guidance_scale", 2.0)

        with torch.inference_mode():
            result = self._pipe(
                prompt="a photo of a model wearing the garment",
                image=person,
                mask_image=agnostic_mask,
                garment_image=garment,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]

        return result
