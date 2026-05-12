from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    garment_path: Path
    pose_id: str
    pose_path: Path

    # filled in by the preprocessor
    garment_category: str = ""

    # filled in after completion
    status: JobStatus = JobStatus.PENDING
    output_path: Path | None = None
    error: str = ""

    @property
    def product_id(self) -> str:
        return self.garment_path.stem

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "garment_path": str(self.garment_path),
            "pose_id": self.pose_id,
            "garment_category": self.garment_category,
            "status": self.status.value,
            "output_path": str(self.output_path) if self.output_path else None,
            "error": self.error,
        }
