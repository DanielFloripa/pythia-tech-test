"""
Pydantic contract for the Pythia prediction service.

Design decisions reflected here (ver design-doc.md):
- Segment state is granular per pipeline stage, not binary success/failed —
  gives the caller visibility into *what* is happening / *where* it failed.
- Job-level status is derived from segment states, never set independently
  (avoids drift between job.status and segments[*].status).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """One stage of the per-segment pipeline, in execution order.

    Used both to report progress (segment.current_stage while running) and
    to report where a failure happened (segment.failed_stage).
    """
    FETCHING_PLAYERS = "fetching_players"
    FETCHING_REVENUES = "fetching_revenues"
    FETCHING_GAMEROUNDS = "fetching_gamerounds"
    FITTING_PLAYERS = "fitting_players"
    FITTING_ECONOMY = "fitting_economy"
    FITTING_GAMEROUNDS = "fitting_gamerounds"
    PREDICTING = "predicting"


class SegmentStatus(str, Enum):
    PENDING = "pending"       # queued, worker not picked it up yet
    RUNNING = "running"       # worker is executing; see current_stage
    SUCCESS = "success"
    FAILED = "failed"         # see failed_stage + error_message


class JobStatus(str, Enum):
    PENDING = "pending"                 # no segment started yet
    RUNNING = "running"                 # at least one segment still pending/running
    SUCCESS = "success"                 # all segments succeeded
    PARTIAL_SUCCESS = "partial_success"  # mixed success/failed, none pending/running
    FAILED = "failed"                   # all segments failed


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    games: List[str] = Field(..., min_length=1, description="Game IDs to predict for")
    regions: List[str] = Field(..., min_length=1, description="Regions to predict for")
    platforms: List[str] = Field(..., min_length=1, description="Platforms to predict for")

    @field_validator("games", "regions", "platforms")
    @classmethod
    def _no_blank_or_dupe(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v]
        if any(not s for s in cleaned):
            raise ValueError("empty values are not allowed")
        # de-dupe preserving order — also stabilizes the dedup-key hash downstream
        seen = dict.fromkeys(cleaned)
        return list(seen)


# ---------------------------------------------------------------------------
# Response — segment level
# ---------------------------------------------------------------------------

class SegmentPrediction(BaseModel):
    """Populated only when status == SUCCESS."""
    sum: float
    mean: float


class SegmentResult(BaseModel):
    game: str
    region: str
    platform: str
    status: SegmentStatus

    current_stage: Optional[PipelineStage] = Field(
        default=None, description="Set while status == RUNNING"
    )
    failed_stage: Optional[PipelineStage] = Field(
        default=None, description="Set only when status == FAILED"
    )
    error_message: Optional[str] = Field(
        default=None, description="Set only when status == FAILED"
    )

    prediction: Optional[SegmentPrediction] = Field(
        default=None, description="Set only when status == SUCCESS"
    )

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Response — job level
# ---------------------------------------------------------------------------

class JobSubmitResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    deduplicated: bool = Field(
        description="True if this job_id refers to an already in-flight job "
        "with the same (games, regions, platforms) submitted again"
    )
    segment_count: int
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    segment_count: int
    segments_succeeded: int
    segments_failed: int
    segments: List[SegmentResult]
    created_at: datetime
    updated_at: datetime
