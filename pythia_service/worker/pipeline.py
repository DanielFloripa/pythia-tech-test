"""
One segment, start to finish. This is what gets submitted to the pool.

Inside a segment the seven steps are strictly sequential, same as the DS
script; parallelism lives between segments. The in-segment I/O overlap from
Part 2 of the README stays in its own script, since the README never asks
for it in the backend.

State is written before each step rather than only at the end, which is
affordable because the store is in the same process. A poll landing mid-run
shows the stage the segment is in.

On logs: INFO covers the lifecycle, DEBUG every stage transition, and
queued_s on segment_started is the backpressure number worth watching.
"""

import time
from concurrent.futures import Executor, Future
from datetime import UTC, datetime
from uuid import UUID

from pythia_service.domain.models import (
    PipelineStage,
    SegmentPrediction,
    SegmentResult,
    SegmentStatus,
)
from pythia_service.jobs.store import JobStore, SegmentKey
from pythia_service.logging_setup import get_logger, kv
from pythia_service.worker import pythia_adapter

logger = get_logger("pipeline")


def process_segment(
    store: JobStore, job_id: UUID, segment_key: SegmentKey, enqueued_at: float | None = None
) -> None:
    """Runs the segment and records every transition. Never raises.

    Nobody calls .result() on these futures, so an exception escaping here
    would sit inside a Future that nobody reads, and the segment would look
    RUNNING forever. Catching it and writing FAILED is what makes the
    failure visible at all.

    enqueued_at is a time.monotonic() stamp from submit time, only used to
    report queue wait.
    """
    game, region, platform = segment_key
    segment = f"{game}/{region}/{platform}"
    started_at = datetime.now(UTC)
    t0 = time.monotonic()
    queued_s = round(t0 - enqueued_at, 1) if enqueued_at is not None else None
    logger.info(kv("segment_started", job_id=job_id, segment=segment, queued_s=queued_s))

    def set_stage(stage: PipelineStage) -> None:
        logger.debug(kv("segment_stage", job_id=job_id, segment=segment, stage=stage))
        store.update_segment(
            job_id,
            segment_key,
            SegmentResult(
                game=game,
                region=region,
                platform=platform,
                status=SegmentStatus.RUNNING,
                current_stage=stage,
                started_at=started_at,
            ),
        )

    try:
        set_stage(PipelineStage.FETCHING_PLAYERS)
        players_data = pythia_adapter.fetch_players(game, region, platform)

        set_stage(PipelineStage.FETCHING_REVENUES)
        revenues_data = pythia_adapter.fetch_revenues(game, region, platform)

        set_stage(PipelineStage.FETCHING_GAMEROUNDS)
        gamerounds_data = pythia_adapter.fetch_gamerounds(game, region, platform)

        set_stage(PipelineStage.FITTING_PLAYERS)
        modeled_players = pythia_adapter.fit_players(players_data)

        set_stage(PipelineStage.FITTING_ECONOMY)
        modeled_economy = pythia_adapter.fit_economy(revenues_data, modeled_players)

        set_stage(PipelineStage.FITTING_GAMEROUNDS)
        modeled_gamerounds = pythia_adapter.fit_gamerounds(gamerounds_data, modeled_economy)

        set_stage(PipelineStage.PREDICTING)
        prediction = pythia_adapter.run_prediction(modeled_players, modeled_economy, modeled_gamerounds)

        result = SegmentPrediction(sum=float(prediction.sum()), mean=float(prediction.mean()))
        logger.info(kv(
            "segment_succeeded", job_id=job_id, segment=segment,
            duration_s=round(time.monotonic() - t0, 1), sum=result.sum, mean=result.mean,
        ))
        _record(
            store,
            job_id,
            segment_key,
            SegmentResult(
                game=game,
                region=region,
                platform=platform,
                status=SegmentStatus.SUCCESS,
                prediction=result,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            ),
        )

    except Exception as exc:
        # Where were we? The stage we last wrote to the store, rather than a
        # second variable tracked in parallel through the try block.
        current = store.get_job(job_id)
        failed_stage = current.segments[segment_key].current_stage if current else None
        error = _describe(exc)
        # The API says what failed; only the traceback says why.
        logger.error(
            kv("segment_failed", job_id=job_id, segment=segment, stage=failed_stage,
               duration_s=round(time.monotonic() - t0, 1), error=error),
            exc_info=True,
        )
        _record(
            store,
            job_id,
            segment_key,
            SegmentResult(
                game=game,
                region=region,
                platform=platform,
                status=SegmentStatus.FAILED,
                failed_stage=failed_stage,
                error_message=error,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            ),
        )


def submit_job_segments(
    store: JobStore, pool_executor: Executor, job_id: UUID, segment_keys: list[SegmentKey]
) -> None:
    """Enqueue a job's segments. Fire and forget: the API responds right
    after this, and progress is only ever seen through GET.
    """
    scheduled: list[tuple[SegmentKey, Future]] = []
    for index, segment_key in enumerate(segment_keys):
        try:
            future = pool_executor.submit(process_segment, store, job_id, segment_key, time.monotonic())
        except RuntimeError:
            # The pool went down mid-loop. Anything left pending would strand
            # the job forever and, worse, keep its dedup key occupied, so
            # every identical request afterwards would join a job that can
            # never finish. Deleting the job is not an option either: a
            # deduplicated caller may already hold this id, and their next
            # GET would 404.
            #
            # cancel() tells the two cases apart. It succeeds only for
            # segments still waiting in the queue, which the pool's shutdown
            # would drop anyway; segments already running keep going and
            # write their own result.
            cancelled = [key for key, pending in scheduled if pending.cancel()]
            stranded = cancelled + segment_keys[index:]
            logger.warning(kv(
                "job_schedule_failed", job_id=job_id, stranded=len(stranded),
                still_running=len(scheduled) - len(cancelled),
            ))
            _fail_stranded(store, job_id, stranded)
            raise
        scheduled.append((segment_key, future))


def _fail_stranded(store: JobStore, job_id: UUID, segment_keys: list[SegmentKey]) -> None:
    now = datetime.now(UTC)
    for game, region, platform in segment_keys:
        _record(
            store,
            job_id,
            (game, region, platform),
            SegmentResult(
                game=game,
                region=region,
                platform=platform,
                status=SegmentStatus.FAILED,
                error_message="not scheduled: service shutting down",
                finished_at=now,
            ),
        )


def _record(store: JobStore, job_id: UUID, segment_key: SegmentKey, result: SegmentResult) -> None:
    """Write a terminal result; log the job outcome if this write closed it."""
    if not store.update_segment(job_id, segment_key, result):
        return
    job = store.get_job(job_id)
    if job is None:
        return
    segments = job.segments.values()
    logger.info(kv(
        "job_finished", job_id=job_id, status=job.status, segments=len(segments),
        succeeded=sum(s.status == SegmentStatus.SUCCESS for s in segments),
        failed=sum(s.status == SegmentStatus.FAILED for s in segments),
        duration_s=round((job.updated_at - job.created_at).total_seconds(), 1),
    ))


def _describe(exc: Exception) -> str:
    """Plain "KeyError: message". str() on a KeyError adds its own quotes."""
    if len(exc.args) == 1 and isinstance(exc.args[0], str):
        return f"{type(exc).__name__}: {exc.args[0]}"
    return f"{type(exc).__name__}: {exc}"
