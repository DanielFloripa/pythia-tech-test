"""
Segment pipeline: the function submitted to SegmentPool for each
(game, region, platform). One call here = one segment, start to finish.

Scope note: this implements the v1 pipeline (Part 1 of the challenge —
the HTTP backend). It is sequential per-KPI within a segment, matching
scripts/pythia-prediction-v1.py's pythia_oracle. Concurrency happens across
segments (via SegmentPool's thread pool), not within one. The I/O-overlap
optimization described in Part 2 of the README is a separate, standalone
script (scripts/pythia-prediction-v2-optimized.py) that the README does not
ask to be wired into the backend — so it isn't, here.

Because the pool is a ThreadPoolExecutor (see worker/pool.py), this function
runs in the same process and memory space as the JobStore, so it updates
segment state directly and granularly — current_stage before each call,
so a GET mid-flight reflects real progress, not just the final outcome.
"""

from datetime import datetime, timezone
from uuid import UUID

from pythia_service.domain.models import (
    PipelineStage,
    SegmentPrediction,
    SegmentResult,
    SegmentStatus,
)
from pythia_service.jobs.store import JobStore, SegmentKey
from pythia_service.worker import pythia_adapter


def process_segment(store: JobStore, job_id: UUID, segment_key: SegmentKey) -> None:
    """Run one segment's full pipeline and write every state transition to
    the store. Never raises — any failure from the adapter is caught here
    and turned into a FAILED SegmentResult, because this function runs
    inside a pool thread with no caller waiting on its return value
    (submitted via executor.submit, not awaited inline); an uncaught
    exception here would only surface silently in the Future object and
    never reach the store, leaving the segment stuck at its last state
    forever.
    """
    game, region, platform = segment_key
    started_at = datetime.now(timezone.utc)

    def set_stage(stage: PipelineStage) -> None:
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

        store.update_segment(
            job_id,
            segment_key,
            SegmentResult(
                game=game,
                region=region,
                platform=platform,
                status=SegmentStatus.SUCCESS,
                prediction=SegmentPrediction(sum=float(prediction.sum()), mean=float(prediction.mean())),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            ),
        )

    except Exception as exc:  # noqa: BLE001 — intentionally broad, see docstring
        # current_stage of the in-flight SegmentResult tells us where we were
        # when it broke; re-derive it from what we last set rather than
        # threading a "current stage" variable through the try block twice.
        current = store.get_job(job_id)
        failed_stage = current.segments[segment_key].current_stage if current else None
        store.update_segment(
            job_id,
            segment_key,
            SegmentResult(
                game=game,
                region=region,
                platform=platform,
                status=SegmentStatus.FAILED,
                failed_stage=failed_stage,
                error_message=str(exc),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            ),
        )


def submit_job_segments(store: JobStore, pool_executor, job_id: UUID, segment_keys: list[SegmentKey]) -> None:
    """Enqueue every segment of a job onto the shared pool. Fire-and-forget
    from the caller's point of view — the API returns immediately after
    this; segment completion is only ever observed via GET /predictions/{id}.
    """
    for segment_key in segment_keys:
        pool_executor.submit(process_segment, store, job_id, segment_key)
