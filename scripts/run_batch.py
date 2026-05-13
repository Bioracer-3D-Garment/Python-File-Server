#!/usr/bin/env python3
"""
Main entry point for a batch run.

Usage:
    python scripts/run_batch.py
    python scripts/run_batch.py --config config/pipeline.yaml
    python scripts/run_batch.py --run-id season_2026_ss
    python scripts/run_batch.py --async              # dispatch to Celery queue
    python scripts/run_batch.py --no-open            # skip auto-opening results

Requires FASHN_API_KEY environment variable.
For --async mode, also requires a running Redis broker and Celery worker.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import config as cfg_module
from pipeline.batch_runner import run_batch, run_batch_async
from pipeline.vton.fashn_api import FashnAPIAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _open_results(output_dir: Path) -> None:
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
    parser.add_argument("--config", default=None, help="Path to pipeline.yaml (default: config/pipeline.yaml)")
    parser.add_argument("--run-id", default=None, help="Human-readable run identifier")
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Dispatch jobs to Celery queue (requires Redis + worker)",
    )
    parser.add_argument("--no-open", action="store_true", help="Skip auto-opening result images")
    args = parser.parse_args()

    cfg = cfg_module.load(args.config) if args.config else cfg_module.load()

    if args.async_mode:
        output_dir, tasks = run_batch_async(cfg, run_id=args.run_id)
        logger.info("Dispatched %d tasks. Run output: %s", len(tasks), output_dir)
        logger.info("Monitor progress with: celery -A pipeline.worker inspect active")
    else:
        adapter = FashnAPIAdapter(cfg)
        logger.info("Using adapter: %s", adapter)
        output_dir = run_batch(cfg, adapter, run_id=args.run_id)
        logger.info("Results written to: %s", output_dir)
        if not args.no_open:
            _open_results(output_dir)


if __name__ == "__main__":
    main()
