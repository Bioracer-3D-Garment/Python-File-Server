from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "pipeline.yaml"


def load(path: Path | str = _DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative directory paths against the project root (parent of config/)
    project_root = path.parent.parent
    for key in ("input_dir", "poses_dir", "cache_dir", "output_dir"):
        if key in cfg.get("pipeline", {}):
            p = Path(cfg["pipeline"][key])
            if not p.is_absolute():
                cfg["pipeline"][key] = str(project_root / p)

    # Allow env vars to override leaf string values via PIPELINE__SECTION__KEY
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key in list(values.keys()):
            env_key = f"PIPELINE__{section.upper()}__{key.upper()}"
            if env_key in os.environ:
                values[key] = os.environ[env_key]

    return cfg


def get_adapter_name(cfg: dict[str, Any]) -> str:
    return cfg["vton"]["adapter"]
