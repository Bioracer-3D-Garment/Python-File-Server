from __future__ import annotations

import pytest
from PIL import Image

from pipeline.worker import run_vton_job


def test_run_vton_job_success(tmp_path, sample_config, tiny_rgb, monkeypatch):
    garment_path = tmp_path / "jersey_001.png"
    tiny_rgb.save(garment_path)

    pose_path = tmp_path / "pose_test.png"
    tiny_rgb.save(pose_path)

    cfg = dict(sample_config)
    cfg["pipeline"]["output_dir"] = str(tmp_path / "outputs")

    monkeypatch.setenv("FASHN_API_KEY", "test_key")
    monkeypatch.setattr("pipeline.preprocess.garment.remove_background", lambda img: img.convert("RGBA"))
    monkeypatch.setattr("pipeline.preprocess.garment.detect_category", lambda img: "upper_body")
    monkeypatch.setattr(
        "pipeline.vton.fashn_api.FashnAPIAdapter.generate",
        lambda self, garment, person, category: Image.new("RGB", (32, 32), (100, 100, 100)),
    )

    result = run_vton_job.apply(args=[str(garment_path), str(pose_path), "pose_test", cfg])

    assert result.successful()
    assert result.result["status"] == "done"
    assert result.result["pose_id"] == "pose_test"
    assert result.result["product_id"] == "jersey_001"


def test_run_vton_job_validation_error_no_retry(tmp_path, sample_config, monkeypatch):
    """Unsupported format fails immediately without retry (status=failed in result)."""
    bad_path = tmp_path / "image.tiff"
    bad_path.write_bytes(b"not an image")

    cfg = dict(sample_config)
    cfg["pipeline"]["output_dir"] = str(tmp_path / "outputs")

    monkeypatch.setenv("FASHN_API_KEY", "test_key")

    result = run_vton_job.apply(args=[str(bad_path), str(bad_path), "pose_test", cfg])

    assert result.successful()  # task completed (caught ValueError internally)
    assert result.result["status"] == "failed"
    assert "Unsupported format" in result.result["error"]
