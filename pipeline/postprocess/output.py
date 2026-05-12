from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.job import Job

logger = logging.getLogger(__name__)

try:
    from skimage.metrics import structural_similarity as ssim
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False


def _quality_check(img: Image.Image, threshold: float) -> bool:
    """
    Return True if the image passes the quality gate.
    Checks:
      1. Not mostly black (mean pixel < 10).
      2. SSIM vs. a blank white image is above threshold (detects artifacts).
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)

    if arr.mean() < 10:
        logger.warning("Quality gate: image is nearly black.")
        return False

    if _SSIM_AVAILABLE and threshold > 0:
        h, w = arr.shape[:2]
        center = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        white = np.full_like(center, 255)
        score = ssim(center, white, channel_axis=2, data_range=255)
        # A very high SSIM against white means the output is blank
        if score > 0.95:
            logger.warning("Quality gate: output too similar to blank white (SSIM=%.3f).", score)
            return False

    return True


def save_output(result: Image.Image, job: Job, cfg: dict[str, Any], output_dir: Path | None = None) -> Path:
    """
    Run the quality gate and write the result image to the output directory.
    Raises RuntimeError if the quality gate fails.
    
    Args:
        result: the generated image
        job: job metadata (product_id, pose_id, etc.)
        cfg: config dictionary
        output_dir: explicit output directory (e.g. outputs/run_<timestamp>/). If None, falls back to cfg.
    """
    pcfg = cfg.get("postprocess", {})
    threshold = pcfg.get("quality_threshold", 0.10)
    fmt = pcfg.get("output_format", "PNG").upper()

    if not _quality_check(result, threshold):
        raise RuntimeError(f"Quality gate failed for {job.product_id} × {job.pose_id}")

    # Use provided output_dir or fall back to config
    if output_dir is None:
        output_dir = Path(cfg["pipeline"]["output_dir"])
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save directly in output_dir with product_id + pose_id in filename
    filename = f"{job.product_id}__{job.pose_id}.{fmt.lower()}"
    out_path = output_dir / filename
    result.save(out_path, format=fmt)
    return out_path
