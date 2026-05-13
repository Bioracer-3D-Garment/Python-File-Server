from __future__ import annotations

import pytest
from PIL import Image

from pipeline.batch_runner import run_batch


class _PassthroughAdapter:
    def generate(self, garment, person, category):
        return Image.new("RGB", (32, 32), (100, 100, 100))


class _FailingAdapter:
    def generate(self, garment, person, category):
        raise RuntimeError("Simulated adapter failure")


def _make_batch_env(tmp_path, sample_config, tiny_rgb) -> dict:
    """Wire up a minimal batch environment: 1 garment + 1 pose."""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    tiny_rgb.save(inputs_dir / "jersey_001.png")

    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    tiny_rgb.save(poses_dir / "pose_test.png")

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    cfg = dict(sample_config)
    cfg["pipeline"] = dict(sample_config["pipeline"])
    cfg["pipeline"]["input_dir"] = str(inputs_dir)
    cfg["pipeline"]["poses_dir"] = str(poses_dir)
    cfg["pipeline"]["output_dir"] = str(outputs_dir)

    return cfg


def test_batch_runs_and_writes_results(tmp_path, sample_config, tiny_rgb, monkeypatch):
    cfg = _make_batch_env(tmp_path, sample_config, tiny_rgb)

    monkeypatch.setattr("pipeline.preprocess.garment.remove_background", lambda img: img.convert("RGBA"))
    monkeypatch.setattr("pipeline.preprocess.garment.detect_category", lambda img: "upper_body")

    adapter = _PassthroughAdapter()
    output_dir = run_batch(cfg, adapter, run_id="test_run")

    assert output_dir.exists()
    images = list(output_dir.glob("*.png"))
    assert len(images) == 1  # 1 garment × 1 pose
    assert images[0].exists()
    assert "jersey_001" in images[0].name
    assert "pose_test" in images[0].name


def test_batch_logs_failure(tmp_path, sample_config, tiny_rgb, monkeypatch):
    cfg = _make_batch_env(tmp_path, sample_config, tiny_rgb)

    monkeypatch.setattr("pipeline.preprocess.garment.remove_background", lambda img: img.convert("RGBA"))
    monkeypatch.setattr("pipeline.preprocess.garment.detect_category", lambda img: "upper_body")

    adapter = _FailingAdapter()
    output_dir = run_batch(cfg, adapter, run_id="test_fail")

    images = list(output_dir.glob("*.png"))
    assert len(images) == 0  # adapter failed, no output images
