from __future__ import annotations

import pytest
from PIL import Image

from pipeline.preprocess.pose import load_pose_cache


def test_load_pose_cache_returns_expected_keys(pose_cache):
    cache_dir, pose_id = pose_cache
    result = load_pose_cache(pose_id, cache_dir)
    assert "keypoints" in result
    assert "person" in result
    assert "agnostic_mask" in result
    assert isinstance(result["person"], Image.Image)
    assert isinstance(result["agnostic_mask"], Image.Image)


def test_load_pose_cache_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="(?i)pose cache incomplete"):
        load_pose_cache("nonexistent_pose", tmp_path)
