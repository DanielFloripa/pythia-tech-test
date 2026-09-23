"""
Submit and poll. The handlers validate, hand work to the store and the pool,
and map what comes back onto the response models. No pipeline code here.

They are async because nothing in them waits: a hash, a lock held for
microseconds around two dict operations, and a submit that only enqueues.
Note the lock is a threading.Lock and not asyncio.Lock, since pool threads
take the same lock and asyncio primitives are not thread-safe.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from pythia_service.api.dependencies import PoolDep, StoreDep
from pythia_service.domain.dedup import compute_dedup_key
from pythia_service.domain.models import (
    JobListResponse,
    JobStatus,
    JobStatusResponse,
    JobSubmitResponse,
    JobSummary,
    PredictionRequest,
    SegmentStatus,
)
from pythia_service.jobs.store import Job
from pythia_service.logging_setup import get_logger, kv
from pythia_service.worker.pipeline import submit_job_segments

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = get_logger("api")


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=JobSubmitResponse)
async def submit_prediction(body: PredictionRequest, store: StoreDep, pool: PoolDep) -> JobSubmitResponse:
    # Read before anything is created: no point registering a job that has
    # nowhere to run.
    try:
        executor = pool.executor
    except RuntimeError as exc:
        logger.warning(kv("job_rejected", reason="shutting_down"))
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "service is shutting down") from exc

    segment_keys = body.segment_keys()
    dedup_key = compute_dedup_key(body)
    job, created = store.get_or_create_job(body, dedup_key, segment_keys)
    if created:
        logger.info(kv("job_accepted", job_id=job.job_id, segments=len(segment_keys), dedup_key=dedup_key[:12]))
    else:
        logger.info(kv("job_deduplicated", job_id=job.job_id, status=job.status, dedup_key=dedup_key[:12]))

    # Only the creator schedules. Scheduling on a dedup hit would run every
    # segment twice, with two threads writing the same keys: a finished
    # segment could go back to RUNNING.
    if created:
        try:
            submit_job_segments(store, executor, job.job_id, segment_keys)
        except RuntimeError as exc:
            # Shutdown beat us between the check above and the submit.
            # submit_job_segments already marked the rest FAILED, so the job
            # is terminal and still answers a poll.
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"service is shutting down; job {job.job_id} could not be fully scheduled",
            ) from exc

    # Snapshot from creation time, so this usually reads "pending" even if a
    # thread has already picked a segment up. Point-in-time by design.
    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        deduplicated=not created,
        segment_count=len(job.segments),
        created_at=job.created_at,
    )


@router.get("", response_model=JobListResponse)
async def list_predictions(
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    status_filter: Annotated[JobStatus | None, Query(alias="status")] = None,
) -> JobListResponse:
    """Newest first, without segments. Not required by the brief, but without
    it the only way to ask "what is the service working on" is to read logs.
    The limit is mandatory because the store never forgets a job.
    """
    jobs = store.list_jobs(limit=limit, status=status_filter)
    return JobListResponse(jobs=[_to_summary(job) for job in jobs], returned=len(jobs))


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_prediction(job_id: UUID, store: StoreDep) -> JobStatusResponse:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    return _to_status_response(job)


def _to_summary(job: Job) -> JobSummary:
    segments = job.segments.values()
    return JobSummary(
        job_id=job.job_id,
        status=job.status,
        segment_count=len(segments),
        segments_succeeded=sum(s.status == SegmentStatus.SUCCESS for s in segments),
        segments_failed=sum(s.status == SegmentStatus.FAILED for s in segments),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _to_status_response(job: Job) -> JobStatusResponse:
    # One snapshot feeds every field, so the counters cannot disagree with
    # the list they are counting.
    segments = list(job.segments.values())
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        segment_count=len(segments),
        segments_succeeded=sum(s.status == SegmentStatus.SUCCESS for s in segments),
        segments_failed=sum(s.status == SegmentStatus.FAILED for s in segments),
        segments=segments,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
