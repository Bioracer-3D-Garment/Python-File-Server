from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.utils.image import pad_to_aspect

"""
Pose preprocessing — runs once per pose image and caches results.

Cache layout for each pose (e.g. poses/pose_001.jpg):
  cache/pose_001_keypoints.json   ← DWPose keypoints
  cache/pose_001_parse.png        ← human parsing segmentation (palette mode)
  cache/pose_001_agnostic.png     ← agnostic mask (inpainting region)

DWPose and human parsing (Graphonomy / SCHP) must be installed separately.
See README for setup instructions.
"""

_DWPOSE = None
_PARSER = None


def _get_dwpose():
    global _DWPOSE
    if _DWPOSE is None:
        import os, sys
        dwpose_path = os.environ.get("DWPOSE_PATH", "DWPose")
        sys.path.insert(0, dwpose_path)
        from dwpose import DWposeDetector  # type: ignore
        _DWPOSE = DWposeDetector()
    return _DWPOSE


def _get_parser():
    global _PARSER
    if _PARSER is None:
        import os, sys
        schp_path = os.environ.get("SCHP_PATH", "SCHP")
        sys.path.insert(0, schp_path)
        from simple_human_parser import HumanParser  # type: ignore
        _PARSER = HumanParser()
    return _PARSER


def _agnostic_mask_from_parse(parse: Image.Image, category: str) -> Image.Image:
    """
    Build an agnostic mask (white = region to inpaint) from a parsing map.
    Pixel labels follow the ATR/LIP palette used by SCHP/Graphonomy.
    """
    arr = np.array(parse)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)

    if category in ("upper_body", "dresses"):
        # upper clothing labels: 5=upper-clothes, 7=coat, 8=jumpsuits, 11=dress
        for lbl in [5, 7, 8, 11]:
            mask[arr == lbl] = 255
    if category in ("lower_body", "dresses"):
        # lower clothing labels: 6=pants, 12=skirt
        for lbl in [6, 12]:
            mask[arr == lbl] = 255

    return Image.fromarray(mask, mode="L")


def preprocess_pose(
    pose_path: Path,
    cache_dir: Path,
    cfg: dict[str, Any],
    category: str = "upper_body",
    force: bool = False,
) -> dict[str, Any]:
    """
    Run DWPose + human parsing on a pose image and cache the results.
    Subsequent calls load from cache unless force=True.

    Returns a dict with keys: keypoints, parse_path, agnostic_path, person_path.
    """
    pose_id = pose_path.stem
    kp_path = cache_dir / f"{pose_id}_keypoints.json"
    parse_path = cache_dir / f"{pose_id}_parse.png"
    agnostic_path = cache_dir / f"{pose_id}_agnostic.png"
    person_path = cache_dir / f"{pose_id}_person.png"

    gcfg = cfg.get("garment", {})
    w = gcfg.get("target_width", 768)
    h = gcfg.get("target_height", 1024)

    if not force and all(p.exists() for p in [kp_path, parse_path, agnostic_path, person_path]):
        with open(kp_path) as f:
            keypoints = json.load(f)
        return {
            "keypoints": keypoints,
            "parse_path": parse_path,
            "agnostic_path": agnostic_path,
            "person_path": person_path,
        }

    person = Image.open(pose_path).convert("RGB")
    person_resized = pad_to_aspect(person, w, h)
    person_resized.save(person_path)

    try:
        dwpose = _get_dwpose()
        keypoints = dwpose(person_resized)
    except Exception:
        keypoints = {}
    with open(kp_path, "w") as f:
        json.dump(keypoints, f)

    try:
        parser = _get_parser()
        parse = parser(person_resized)
    except Exception:
        parse = Image.new("P", (w, h), 0)
    parse.save(parse_path)

    agnostic = _agnostic_mask_from_parse(parse, category)
    agnostic.save(agnostic_path)

    return {
        "keypoints": keypoints,
        "parse_path": parse_path,
        "agnostic_path": agnostic_path,
        "person_path": person_path,
    }


def load_pose_cache(pose_id: str, cache_dir: Path) -> dict[str, Any]:
    """Load pre-built pose cache for a given pose_id."""
    kp_path = cache_dir / f"{pose_id}_keypoints.json"
    parse_path = cache_dir / f"{pose_id}_parse.png"
    agnostic_path = cache_dir / f"{pose_id}_agnostic.png"
    person_path = cache_dir / f"{pose_id}_person.png"

    # Include parse_path in the completeness check so callers get a full
    # picture of which cache files are missing (parse was previously
    # omitted from the reported list).
    missing = [p for p in [kp_path, parse_path, agnostic_path, person_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Pose cache incomplete for '{pose_id}'. Missing: {missing}. "
            "Run: python scripts/preprocess_poses.py"
        )

    with open(kp_path) as f:
        keypoints = json.load(f)

    return {
        "keypoints": keypoints,
        "parse_path": parse_path,
        "agnostic_path": agnostic_path,
        "person_path": person_path,
        "person": Image.open(person_path).convert("RGB"),
        "agnostic_mask": Image.open(agnostic_path).convert("L"),
    }
