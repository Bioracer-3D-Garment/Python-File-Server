from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from pipeline.vton.base import VTONAdapter


class NanoAdapter(VTONAdapter):
    """
    Compositing-based placeholder — no API keys, no GPU, no external model.

    Crops the garment to its content region, fits it inside the bounding box
    of the agnostic mask on the person image, then alpha-composites the result.

    TODO: replace _composite() with a lightweight model inference call once
    the target model is chosen (e.g. a small diffusion or GAN model).
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

    def generate(
        self,
        garment: Image.Image,
        garment_mask: Image.Image,
        person: Image.Image,
        agnostic_mask: Image.Image,
        pose_data: dict[str, Any],
        category: str,
    ) -> Image.Image:
        # Pass the precomputed garment_mask through to the compositor so we
        # use a reliable alpha/mask when placing the garment. This improves
        # results vs. relying on the source image alpha channel alone.
        return self._composite(garment, person, agnostic_mask, category, garment_mask)

    def _composite(
        self,
        garment: Image.Image,
        person: Image.Image,
        agnostic_mask: Image.Image,
        category: str = "upper_body",
        garment_mask: Image.Image | None = None,
    ) -> Image.Image:
        mask_arr = np.array(agnostic_mask.convert("L"))
        rows = np.any(mask_arr > 128, axis=1)
        cols = np.any(mask_arr > 128, axis=0)
        if not rows.any():
            # No pose model available — fall back to a category-based region estimate
            return self._composite_estimated(garment, person, category)

        r_min = int(rows.argmax())
        r_max = int(len(rows) - rows[::-1].argmax() - 1)
        c_min = int(cols.argmax())
        c_max = int(len(cols) - cols[::-1].argmax() - 1)
        target_w = c_max - c_min
        target_h = r_max - r_min
        if target_w <= 0 or target_h <= 0:
            return person.convert("RGB")

        # Prefer the explicit garment_mask (produced by preprocess_garment)
        # when available — it tends to be cleaner and more reliable than the
        # original image alpha channel. Resize mask if necessary to match
        # the garment image.
        garment_rgba = garment.convert("RGBA")
        if garment_mask is not None:
            gm = garment_mask.convert("L")
            if gm.size != garment_rgba.size:
                gm = gm.resize(garment_rgba.size, resample=Image.NEAREST)
            # Feather the mask slightly for smoother compositing.
            try:
                feather_radius = int(self._cfg.get("vton", {}).get("feather_radius", 3))
            except Exception:
                feather_radius = 3
            if feather_radius > 0:
                gm = gm.filter(ImageFilter.GaussianBlur(radius=feather_radius))
            alpha_arr = np.array(gm)
        else:
            alpha_arr = np.array(garment_rgba)[:, :, 3]

        g_rows = np.any(alpha_arr > 10, axis=1)
        g_cols = np.any(alpha_arr > 10, axis=0)
        if g_rows.any() and g_cols.any():
            gr_min = int(g_rows.argmax())
            gr_max = int(len(g_rows) - g_rows[::-1].argmax() - 1)
            gc_min = int(g_cols.argmax())
            gc_max = int(len(g_cols) - g_cols[::-1].argmax() - 1)
            garment_rgba = garment_rgba.crop((gc_min, gr_min, gc_max + 1, gr_max + 1))

        garment_rgba = _fit(garment_rgba, target_w, target_h)

        result = person.convert("RGBA")
        paste_x = c_min + (target_w - garment_rgba.width) // 2
        paste_y = r_min + (target_h - garment_rgba.height) // 2
        result.paste(garment_rgba, (paste_x, paste_y), garment_rgba.split()[3])
        return result.convert("RGB")


    def _composite_estimated(
        self,
        garment: Image.Image,
        person: Image.Image,
        category: str,
    ) -> Image.Image:
        """Fallback used when no pose model produced an agnostic mask."""
        w, h = person.size
        # Rough torso/body regions as fractions of image dimensions
        regions = {
            "upper_body": (0.15, 0.15, 0.85, 0.65),
            "lower_body": (0.15, 0.50, 0.85, 0.92),
            "dresses":    (0.15, 0.15, 0.85, 0.92),
        }
        x0f, y0f, x1f, y1f = regions.get(category, regions["upper_body"])
        fake_mask = Image.new("L", (w, h), 0)
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(fake_mask)
        draw.rectangle([int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h)], fill=255)
        return self._composite(garment, person, fake_mask, category)


def _fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)
