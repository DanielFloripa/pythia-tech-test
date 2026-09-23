"""
Wire contract for the prediction service.

Two things worth knowing before reading: segment state is per stage rather
than a success/failed flag, so a caller can see where a job actually is, and
job status is never stored, only derived (see jobs/state.py).
"""

from __future__ import annotations

import os
from datetime import datetime
from enum import Enum
from itertools import product
from typing import List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 8x today's catalog. Caps the worst case at roughly 14 min of machine time.
# The right number is a product call, hence the env var.
MAX_SEGMENTS_PER_REQUEST = int(os.environ.get("PYTHIA_MAX_SEGMENTS", "64"))


class PipelineStage(str, Enum):
    """The seven steps of one segment, in execution order."""
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
        return list(dict.fromkeys(cleaned))  # de-dupe, keep order

    @model_validator(mode="after")
    def _bounded_fan_out(self) -> PredictionRequest:
        # A cartesian product: 50x50x50 strings is 125k segments, weeks of
        # compute, and the pool is FIFO. Bounds one request, not total load.
        count = len(self.games) * len(self.regions) * len(self.platforms)
        if count > MAX_SEGMENTS_PER_REQUEST:
            raise ValueError(
                f"request expands to {count} segments; the limit is {MAX_SEGMENTS_PER_REQUEST}"
            )
        return self

    def segment_keys(self) -> List[Tuple[str, str, str]]:
        """One (game, region, platform) tuple per segment, stable order."""
        return list(product(self.games, self.regions, self.platforms))


class SegmentPrediction(BaseModel):
    """Populated only when status == SUCCESS."""
    sum: float
    mean: float


class SegmentResult(BaseModel):
    """Frozen: the store swaps these objects, it never edits one in place.
    That is what makes the store's shallow snapshot a consistent read, so the
    invariant is enforced here rather than left to discipline.
    """
    model_config = ConfigDict(frozen=True)

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
