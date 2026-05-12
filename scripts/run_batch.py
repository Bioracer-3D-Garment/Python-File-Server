#!/usr/bin/env python3
"""
Main entry point for a batch run.

Usage:
    python scripts/run_batch.py
    python scripts/run_batch.py --adapter fashn_api --config config/pipeline.yaml
    python scripts/run_batch.py --adapter idm_vton --run-id season_2026_ss
    python scripts/run_batch.py --no-open   # skip auto-opening result images

Adapter override via CLI takes precedence over pipeline.yaml.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import config as cfg_module
from pipeline.batch_runner import run_batch
from pipeline.vton.base import build_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _open_results(output_dir: Path) -> None:
    """
    Find all PNG/JPG images in the output directory and open them with xdg-open.
    Silently skips if xdg-open is unavailable or if there are no images.
    """
    try:
        opened = 0
        image_exts = {".png", ".jpg", ".jpeg"}
        for img_file in sorted(output_dir.glob("*")):
            if img_file.suffix.lower() in image_exts and img_file.is_file():
                subprocess.Popen(
                    ["xdg-open", str(img_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                opened += 1
        if opened > 0:
            logger.info("Opened %d result image(s)", opened)
    except Exception as exc:
        logger.debug("Could not auto-open results: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bioracer VTON batch pipeline.")
    parser.add_argument("--config", default=None, help="Path to pipeline.yaml (default: config/pipeline.yaml next to project root)")
    parser.add_argument(
        "--adapter",
        choices=["nano", "fashn_api", "fal_api", "idm_vton", "ootdiffusion"],
        default=None,
        help="Override the adapter from config (default: nano)",
    )
    parser.add_argument("--run-id", default=None, help="Human-readable run identifier")
    parser.add_argument("--no-open", action="store_true", help="Skip auto-opening result images after batch completes")
    args = parser.parse_args()

    cfg = cfg_module.load(args.config) if args.config else cfg_module.load()
    if args.adapter:
        cfg["vton"]["adapter"] = args.adapter

    adapter = build_adapter(cfg)
    logger.info("Using adapter: %s", adapter)

    output_dir = run_batch(cfg, adapter, run_id=args.run_id)
    logger.info("Results written to: %s", output_dir)

    if not args.no_open:
        _open_results(output_dir)


if __name__ == "__main__":
    main()
