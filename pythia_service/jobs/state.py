"""
Job status is computed, never stored. A stored copy would eventually drift:
some update path forgets to refresh it and the job sits "running" with every
segment long finished.

No I/O, no JobStore import, so it stays trivial to test on its own.
"""

from __future__ import annotations

from typing import Sequence

from pythia_service.domain.models import JobStatus, SegmentStatus


def derive_job_status(segment_statuses: Sequence[SegmentStatus]) -> JobStatus:
    """Roll segment statuses up into one job status.

    A terminal status is only reported once nothing is in flight: 3 done and
    2 running is RUNNING, not PARTIAL_SUCCESS, so a caller never treats a
    half-finished job as final.
    """
    if not segment_statuses:
        # Every job is expanded into at least one segment at creation, so an
        # empty list is an upstream bug. Raise instead of quietly returning
        # PENDING. Not an assert: python -O would drop it.
        raise ValueError("cannot derive JobStatus from an empty segment list")

    # Queueing is visible on purpose. With a bounded pool a job can sit for
    # minutes before anything starts, and calling that "running" tells the
    # caller work is happening when none is.
    if all(s == SegmentStatus.PENDING for s in segment_statuses):
        return JobStatus.PENDING

    if any(s in (SegmentStatus.PENDING, SegmentStatus.RUNNING) for s in segment_statuses):
        return JobStatus.RUNNING

    successes = sum(1 for s in segment_statuses if s == SegmentStatus.SUCCESS)
    failures = sum(1 for s in segment_statuses if s == SegmentStatus.FAILED)

    if failures == 0:
        return JobStatus.SUCCESS
    if successes == 0:
        return JobStatus.FAILED
    return JobStatus.PARTIAL_SUCCESS
