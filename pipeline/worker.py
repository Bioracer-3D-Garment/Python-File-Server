from __future__ import annotations

import traceback

from celery import Celery


def make_celery(cfg: dict | None = None) -> Celery:
    if cfg is None:
        try:
            from pipeline.config import load as load_cfg
            cfg = load_cfg()
        except Exception:
            cfg = {}
    cc = cfg.get("celery", {})
    return Celery(
        "bioracer",
        broker=cc.get("broker_url", "redis://localhost:6379/0"),
        backend=cc.get("result_backend", "redis://localhost:6379/0"),
    )


app = make_celery()


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def run_vton_job(
    self,
    garment_path: str,
    pose_path: str,
    pose_id: str,
    cfg: dict,
) -> dict:
    """
    Execute a single garment × pose VTON job.

    Returns job.to_dict() on completion (status is 'done' or 'failed').
    Retries up to 3× on transient errors (API timeouts, network issues).
    ValueError (validation, bad format) fails immediately without retry.
    """
    from pathlib import Path
    from PIL import Image
    from pipeline.job import Job, JobStatus
    from pipeline.preprocess.garment import preprocess_garment
    from pipeline.postprocess.output import save_output
    from pipeline.vton.fashn_api import FashnAPIAdapter

    job = Job(garment_path=Path(garment_path), pose_id=pose_id, pose_path=Path(pose_path))
    job.status = JobStatus.RUNNING
    output_dir = Path(cfg["pipeline"]["output_dir"])
    adapter = FashnAPIAdapter(cfg)

    try:
        garment, category = preprocess_garment(job.garment_path, cfg)
        job.garment_category = category
        person = Image.open(job.pose_path).convert("RGB")
        result = adapter.generate(garment=garment, person=person, category=category)
        out_path = save_output(result, job, cfg, output_dir)
        job.output_path = out_path
        job.status = JobStatus.DONE
        return job.to_dict()
    except ValueError as exc:
        # Permanent failure — validation or format error, no retry
        job.status = JobStatus.FAILED
        job.error = str(exc)
        return job.to_dict()
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = traceback.format_exc()
        raise self.retry(exc=exc)
