from __future__ import annotations

import io

import numpy as np
from PIL import Image


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def pad_to_aspect(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize image to fit within (width, height) preserving aspect ratio, padding with white."""
    img.thumbnail((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    offset = ((width - img.width) // 2, (height - img.height) // 2)
    canvas.paste(img, offset, mask=img if img.mode == "RGBA" else None)
    return canvas


def binary_mask_from_alpha(img: Image.Image, threshold: int = 127) -> Image.Image:
    """Extract binary mask from RGBA alpha channel."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.array(img.split()[3])
    mask = (alpha > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask).convert("L")
