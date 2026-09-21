"""
In-memory JobStore.

Architectural note (why this file looks the way it does):
The segment pool is a ThreadPoolExecutor (see worker/pool.py) — the fit
steps are blocking calls that wait on pythia_library's own internal
subprocess pool, which releases the GIL while waiting, so threads are
sufficient at our layer (see worker/pool.py docstring for the full
reasoning). This means worker code runs in the *same* process and *same*
memory space as this store — no IPC needed to get updates back.

Consequence: pipeline.py can call store.update_segment(...) directly, from
whichever thread is executing that segment, at every stage transition —
current_stage can reflect real-time progress accurately, not just the
terminal outcome. The threading.Lock below exists for the ordinary reason a
shared mutable dict needs one: multiple pool threads updating different (or
the same) job concurrently.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pythia_service.domain.models import (
    JobStatus,
    PredictionRequest,
    SegmentResult,
    SegmentStatus,
)
from pythia_service.jobs.state import derive_job_status

SegmentKey = Tuple[str, str, str]  # (game, region, platform)


@dataclass
class Job:
    job_id: UUID
    dedup_key: str
    request: PredictionRequest
    segments: Dict[SegmentKey, SegmentResult]
    created_at: datetime
    updated_at: datetime

    @property
    def status(self) -> JobStatus:
        return derive_job_status([seg.status for seg in self.segments.values()])

    def is_terminal(self) -> bool:
        return self.status in (
            JobStatus.SUCCESS,
            JobStatus.FAILED,
            JobStatus.PARTIAL_SUCCESS,
        )


class JobStore:
    """Thread-safe in-memory store. Swap point for Postgres/Redis later —
    every method here is the interface the rest of the app depends on, so a
    persistent implementation only needs to satisfy the same signatures.
    """

    def __init__(self) -> None:
        self._jobs: Dict[UUID, Job] = {}
        # dedup_key -> job_id, only for jobs that are NOT terminal yet.
        # A finished job is removed from this index so a later identical
        # request creates a fresh job instead of returning a stale result.
        self._active_by_dedup_key: Dict[str, UUID] = {}
        self._lock = threading.Lock()

    def find_active_by_dedup_key(self, dedup_key: str) -> Optional[Job]:
        with self._lock:
            job_id = self._active_by_dedup_key.get(dedup_key)
            return self._jobs[job_id] if job_id else None

    def create_job(
        self, request: PredictionRequest, dedup_key: str, segment_keys: List[SegmentKey]
    ) -> Job:
        now = datetime.now(timezone.utc)
        job = Job(
            job_id=uuid4(),
            dedup_key=dedup_key,
            request=request,
            segments={
                key: SegmentResult(
                    game=key[0],
                    region=key[1],
                    platform=key[2],
                    status=SegmentStatus.PENDING,
                )
                for key in segment_keys
            },
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._active_by_dedup_key[dedup_key] = job.job_id
        return job

    def get_job(self, job_id: UUID) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_segment(self, job_id: UUID, segment_key: SegmentKey, updated: SegmentResult) -> None:
        """Replace one segment's state. Called directly from whichever pool
        thread is executing that segment — see the module docstring.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return  # defensive: job was never created or was purged
            job.segments[segment_key] = updated
            job.updated_at = datetime.now(timezone.utc)
            if job.is_terminal():
                # free the dedup slot so a future identical request doesn't
                # get silently merged into this already-finished job
                self._active_by_dedup_key.pop(job.dedup_key, None)
