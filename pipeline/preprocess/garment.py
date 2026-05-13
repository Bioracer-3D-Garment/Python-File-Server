from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.utils.image import pad_to_aspect

_REMBG_SESSION = None

SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "webp"}


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session("u2net")
    return _REMBG_SESSION


def remove_background(img: Image.Image) -> Image.Image:
    """Return RGBA image with background removed using rembg (u2net)."""
    from rembg import remove
    return remove(img, session=_get_rembg_session())


def detect_category(img: Image.Image) -> str:
    """
    Detect garment category using CLIP zero-shot classification.
    Returns one of: 'upper_body', 'lower_body', 'dresses'.

    Falls back to 'upper_body' if transformers/CLIP is unavailable.
    """
    try:
        from transformers import pipeline as hf_pipeline
        classifier = hf_pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
        )
        labels = ["upper body clothing", "lower body clothing", "dress or full body outfit"]
        result = classifier(img.convert("RGB"), candidate_labels=labels)
        top = result[0]["label"]
        if "lower" in top:
            return "lower_body"
        if "dress" in top or "full" in top:
            return "dresses"
        return "upper_body"
    except Exception:
        return "upper_body"


def validate_garment_image(path: Path, cfg: dict[str, Any]) -> None:
    """
    Raise ValueError if the garment image fails pre-processing checks.
    Called before any heavy processing (rembg, CLIP) to fail fast.
    """
    gcfg = cfg.get("garment", {})
    max_mb = gcfg.get("max_file_mb", 20)
    min_w = gcfg.get("min_width", 100)
    min_h = gcfg.get("min_height", 100)

    ext = path.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Accepted: {sorted(SUPPORTED_FORMATS)}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"Image too large: {size_mb:.1f} MB (limit {max_mb} MB)")

    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc

    if w < min_w or h < min_h:
        raise ValueError(
            f"Image too small: {w}×{h} px (minimum {min_w}×{min_h})"
        )


def preprocess_garment(
    path: Path,
    cfg: dict[str, Any],
) -> tuple[Image.Image, str]:
    """
    Validate, load, remove background, resize, and categorise a garment image.

    Returns:
        garment_rgb:  RGB image resized and padded to target dimensions
        category:     'upper_body' | 'lower_body' | 'dresses'
    """
    validate_garment_image(path, cfg)

    gcfg = cfg.get("garment", {})
    w = gcfg.get("target_width", 768)
    h = gcfg.get("target_height", 1024)

    raw = Image.open(path).convert("RGBA")
    rgba = remove_background(raw)
    garment_rgb = pad_to_aspect(rgba, w, h)
    category = detect_category(rgba)

    return garment_rgb, category
