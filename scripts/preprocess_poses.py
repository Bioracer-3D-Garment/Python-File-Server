#!/usr/bin/env python3
"""
One-time script: build the pose cache from the poses/ directory.

Run this whenever you add or change pose images.
Results are written to cache/ and reused by every batch run.

Usage:
    python scripts/preprocess_poses.py
    python scripts/preprocess_poses.py --poses-dir poses/ --cache-dir cache/ --category upper_body
    python scripts/preprocess_poses.py --force   # rebuild even if cache exists
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import config as cfg_module
from pipeline.preprocess.pose import preprocess_pose

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess pose images and write to cache.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--poses-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--category", default="upper_body", choices=["upper_body", "lower_body", "dresses"])
    parser.add_argument("--force", action="store_true", help="Rebuild cache even if it already exists")
    args = parser.parse_args()

    cfg = cfg_module.load(args.config) if args.config else cfg_module.load()
    poses_dir = Path(args.poses_dir or cfg["pipeline"]["poses_dir"])
    cache_dir = Path(args.cache_dir or cfg["pipeline"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(p for p in poses_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not pose_files:
        logger.error("No pose images found in %s", poses_dir)
        sys.exit(1)

    logger.info("Processing %d pose(s) → %s", len(pose_files), cache_dir)
    for pose_path in pose_files:
        logger.info("  %s", pose_path.name)
        try:
            preprocess_pose(pose_path, cache_dir, cfg, category=args.category, force=args.force)
            logger.info("  ✓ cached")
        except Exception as exc:
            logger.error("  ✗ failed: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
