from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.job import Job, JobStatus
from pipeline.preprocess.garment import preprocess_garment
from pipeline.preprocess.pose import load_pose_cache
from pipeline.postprocess.output import save_output
from pipeline.vton.base import VTONAdapter

logger = logging.getLogger(__name__)


def _discover_garments(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)


def _discover_poses(poses_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p.stem for p in poses_dir.iterdir() if p.suffix.lower() in exts)


def _run_job(job: Job, adapter: VTONAdapter, cfg: dict[str, Any], output_dir: Path | None = None) -> Job:
    job.status = JobStatus.RUNNING
    try:
        cache_dir = Path(cfg["pipeline"]["cache_dir"])
        garment, garment_mask, category = preprocess_garment(job.garment_path, cfg)
        job.garment_category = category

        pose = load_pose_cache(job.pose_id, cache_dir)

        result: Image.Image = adapter.generate(
            garment=garment,
            garment_mask=garment_mask,
            person=pose["person"],
            agnostic_mask=pose["agnostic_mask"],
            pose_data=pose["keypoints"],
            category=category,
        )

        out_path = save_output(result, job, cfg, output_dir=output_dir)
        job.output_path = out_path
        job.status = JobStatus.DONE
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = traceback.format_exc()
        logger.error("Job failed [%s × %s]: %s", job.product_id, job.pose_id, exc)

    return job


def run_batch(
    cfg: dict[str, Any],
    adapter: VTONAdapter,
    run_id: str | None = None,
) -> Path:
    """
    Process every garment × every pose and write result images directly to outputs/<run_id>/.
    
    Returns the path to the run output directory.
    """
    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    pcfg = cfg["pipeline"]
    input_dir = Path(pcfg["input_dir"])
    poses_dir = Path(pcfg["poses_dir"])
    output_dir = Path(pcfg["output_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    garments = _discover_garments(input_dir)
    pose_ids = _discover_poses(poses_dir)

    if not garments:
        raise ValueError(f"No garment images found in {input_dir}")
    if not pose_ids:
        raise ValueError(f"No pose images found in {poses_dir}")

    jobs = [
        Job(garment_path=g, pose_id=p, pose_path=poses_dir / p)
        for g in garments
        for p in pose_ids
    ]

    logger.info("Starting batch: %d garments × %d poses = %d jobs", len(garments), len(pose_ids), len(jobs))

    workers = pcfg.get("workers", 4)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_job, job, adapter, cfg, output_dir): job for job in jobs}
        for future in as_completed(futures):
            completed_job = future.result()
            logger.info("[%s] %s × %s → %s", completed_job.status.value, completed_job.product_id, completed_job.pose_id, completed_job.output_path or completed_job.error[:80])

    done = sum(1 for j in jobs if j.status == JobStatus.DONE)
    failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
    logger.info("Batch complete: %d done, %d failed. Results: %s", done, failed, output_dir)
    return output_dir
