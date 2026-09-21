"""
Job-level status is always *derived* from segment statuses, never set
independently. This avoids the two ever drifting apart (e.g. a job stuck
"running" after all its segments finished).

Kept as a pure function with no I/O and no dependency on JobStore, so it's
trivially unit-testable and reusable from both the worker (after each
segment update) and the API layer (if it ever needs to recompute on read).
"""

from __future__ import annotations

from typing import Sequence

from pythia_service.domain.models import JobStatus, SegmentStatus


def derive_job_status(segment_statuses: Sequence[SegmentStatus]) -> JobStatus:
    """Aggregate all segment statuses of a job into a single JobStatus.

    Decision: "partial_success" only applies once *no* segment is still
    pending/running — a job with 3 successes and 2 still running is still
    reported as RUNNING, not PARTIAL_SUCCESS, because the caller shouldn't
    treat it as final while work is in flight.
    """
    if not segment_statuses:
        # Defensive: a job must always be expanded into >=1 segment at
        # creation time. Getting here means a bug upstream, not a valid
        # empty-job state — surfaced loudly rather than silently PENDING.
        raise ValueError("cannot derive JobStatus from an empty segment list")

    if any(s in (SegmentStatus.PENDING, SegmentStatus.RUNNING) for s in segment_statuses):
        return JobStatus.RUNNING

    # At this point every segment is terminal (SUCCESS or FAILED).
    successes = sum(1 for s in segment_statuses if s == SegmentStatus.SUCCESS)
    failures = sum(1 for s in segment_statuses if s == SegmentStatus.FAILED)

    if failures == 0:
        return JobStatus.SUCCESS
    if successes == 0:
        return JobStatus.FAILED
    return JobStatus.PARTIAL_SUCCESS
