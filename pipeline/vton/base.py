from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

GARMENT_CATEGORIES = ("upper_body", "lower_body", "dresses")


class VTONAdapter(ABC):
    """
    Swap-safe interface for virtual try-on backends.

    Every adapter receives the same inputs and must return a PIL Image.
    The active adapter is selected via config/pipeline.yaml → vton.adapter.
    """

    @abstractmethod
    def generate(
        self,
        garment: Image.Image,
        garment_mask: Image.Image,
        person: Image.Image,
        agnostic_mask: Image.Image,
        pose_data: dict[str, Any],
        category: str,
    ) -> Image.Image:
        """
        Args:
            garment:       flat-lay garment image (RGBA, bg removed)
            garment_mask:  binary mask of the garment region
            person:        model pose image (RGB)
            agnostic_mask: mask of the region on the model to inpaint
            pose_data:     keypoints dict from DWPose (may be empty for API adapters)
            category:      one of GARMENT_CATEGORIES

        Returns:
            PIL Image of the model wearing the garment.
        """

    def __repr__(self) -> str:
        return self.__class__.__name__


def build_adapter(cfg: dict) -> VTONAdapter:
    name = cfg["vton"]["adapter"]
    if name == "nano":
        from pipeline.vton.nano import NanoAdapter
        return NanoAdapter(cfg)
    if name == "idm_vton":
        from pipeline.vton.idm_vton import IDMVTONAdapter
        return IDMVTONAdapter(cfg)
    if name == "ootdiffusion":
        from pipeline.vton.ootdiffusion import OOTDiffusionAdapter
        return OOTDiffusionAdapter(cfg)
    if name == "fashn_api":
        from pipeline.vton.fashn_api import FashnAPIAdapter
        return FashnAPIAdapter(cfg)
    if name == "fal_api":
        from pipeline.vton.fal_api import FalAPIAdapter
        return FalAPIAdapter(cfg)
    raise ValueError(f"Unknown VTON adapter: {name!r}. Choose from: nano, idm_vton, ootdiffusion, fashn_api, fal_api")
