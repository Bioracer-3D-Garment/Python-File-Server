from __future__ import annotations

import copy
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.job import Job, JobStatus
from pipeline.preprocess.garment import preprocess_garment
from pipeline.postprocess.output import save_output
from pipeline.vton.fashn_api import FashnAPIAdapter

logger = logging.getLogger(__name__)


def _discover_garments(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)


def _discover_poses(poses_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in poses_dir.iterdir() if p.suffix.lower() in exts)


def _run_job(job: Job, adapter: FashnAPIAdapter, cfg: dict[str, Any], output_dir: Path | None = None) -> Job:
    job.status = JobStatus.RUNNING
    try:
        garment, category = preprocess_garment(job.garment_path, cfg)
        job.garment_category = category

        person = Image.open(job.pose_path).convert("RGB")

        result: Image.Image = adapter.generate(
            garment=garment,
            person=person,
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
    adapter: FashnAPIAdapter,
    run_id: str | None = None,
) -> Path:
    """
    Process every garment × every pose synchronously via ThreadPoolExecutor.

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
    pose_paths = _discover_poses(poses_dir)

    if not garments:
        raise ValueError(f"No garment images found in {input_dir}")
    if not pose_paths:
        raise ValueError(f"No pose images found in {poses_dir}")

    jobs = [
        Job(garment_path=g, pose_id=pose_path.stem, pose_path=pose_path)
        for g in garments
        for pose_path in pose_paths
    ]

    logger.info("Starting batch: %d garments × %d poses = %d jobs", len(garments), len(pose_paths), len(jobs))

    workers = pcfg.get("workers", 4)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_job, job, adapter, cfg, output_dir): job for job in jobs}
        for future in as_completed(futures):
            completed_job = future.result()
            logger.info(
                "[%s] %s × %s → %s",
                completed_job.status.value,
                completed_job.product_id,
                completed_job.pose_id,
                completed_job.output_path or completed_job.error[:80],
            )

    done = sum(1 for j in jobs if j.status == JobStatus.DONE)
    failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
    logger.info("Batch complete: %d done, %d failed. Results: %s", done, failed, output_dir)
    return output_dir


def run_batch_async(
    cfg: dict[str, Any],
    run_id: str | None = None,
) -> tuple[Path, list]:
    """
    Dispatch all garment × pose jobs to the Celery queue and return immediately.

    Returns (output_dir, list_of_AsyncResult). Requires a running Celery worker
    and Redis broker (see docker-compose.yml).
    """
    from pipeline.worker import run_vton_job

    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    pcfg = cfg["pipeline"]
    input_dir = Path(pcfg["input_dir"])
    poses_dir = Path(pcfg["poses_dir"])
    output_dir = Path(pcfg["output_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deep-copy so each task gets the resolved output_dir for this run
    run_cfg = copy.deepcopy(cfg)
    run_cfg["pipeline"]["output_dir"] = str(output_dir)

    garments = _discover_garments(input_dir)
    pose_paths = _discover_poses(poses_dir)

    if not garments:
        raise ValueError(f"No garment images found in {input_dir}")
    if not pose_paths:
        raise ValueError(f"No pose images found in {poses_dir}")

    tasks = [
        run_vton_job.delay(str(g), str(p), p.stem, run_cfg)
        for g in garments
        for p in pose_paths
    ]

    logger.info(
        "Dispatched %d tasks to Celery queue (run_id=%s)", len(tasks), run_id
    )
    return output_dir, tasks
