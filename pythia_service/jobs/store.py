"""
Job state, in memory.

Segments run on threads in this same process (see worker/pool.py), so a
worker writes progress straight into this dict instead of shipping it back
over IPC. That is the whole reason per-stage progress is cheap here. The
lock is the ordinary price: several pool threads write concurrently, and the
API reads while they do.

Everything the rest of the app needs is in these four methods, which is the
seam to swap for Redis or Postgres when one process stops being enough.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
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
    def __init__(self) -> None:
        self._jobs: Dict[UUID, Job] = {}
        # dedup_key -> job_id, only for jobs that are NOT terminal yet.
        # A finished job is removed from this index so a later identical
        # request creates a fresh job instead of returning a stale result.
        self._active_by_dedup_key: Dict[str, UUID] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _snapshot(job: Job) -> Job:
        """Point-in-time copy. Call it with the lock held; a copy taken
        outside the lock is as torn as the original. Shallow is enough
        because SegmentResult objects get replaced, never edited.
        """
        return replace(job, segments=dict(job.segments))

    def get_or_create_job(
        self, request: PredictionRequest, dedup_key: str, segment_keys: List[SegmentKey]
    ) -> Tuple[Job, bool]:
        """Check and insert under one lock acquisition, returning
        (job, created). Two separate calls would let two identical
        submissions both see "nothing active" and both create a job.
        """
        # Built outside the lock: only check+insert has to be atomic. Costs a
        # throwaway object whenever the request turns out to be a duplicate.
        candidate = self._new_job(request, dedup_key, segment_keys)
        with self._lock:
            existing_id = self._active_by_dedup_key.get(dedup_key)
            if existing_id is not None:
                return self._snapshot(self._jobs[existing_id]), False
            self._jobs[candidate.job_id] = candidate
            self._active_by_dedup_key[dedup_key] = candidate.job_id
            return self._snapshot(candidate), True

    @staticmethod
    def _new_job(
        request: PredictionRequest, dedup_key: str, segment_keys: List[SegmentKey]
    ) -> Job:
        now = datetime.now(timezone.utc)
        return Job(
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

    def get_job(self, job_id: UUID) -> Optional[Job]:
        """A snapshot, never the live Job: a response reads status, counters
        and the segment list, and those have to agree with each other.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return self._snapshot(job) if job is not None else None

    def update_segment(self, job_id: UUID, segment_key: SegmentKey, updated: SegmentResult) -> bool:
        """Replace one segment's state, from the thread running it.

        True means this write is the one that finished the job. The caller
        reacts to that outside the lock; logging is I/O and the lock must
        never wait on I/O.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False  # defensive: job was never created or was purged
            was_terminal = job.is_terminal()
            job.segments[segment_key] = updated
            job.updated_at = datetime.now(timezone.utc)
            if job.is_terminal():
                # Drop the dedup slot: a later identical request deserves a
                # fresh job, not a merge into this finished one.
                self._active_by_dedup_key.pop(job.dedup_key, None)
                return not was_terminal
            return False
