"""
Queue depth, separate from /health on purpose: readiness answers "can I send
traffic here", these counters answer "why is my job still waiting". A probe
and a dashboard are different consumers with different failure modes.
"""

from fastapi import APIRouter

from pythia_service.api.dependencies import PoolDep, StoreDep
from pythia_service.domain.models import StatsResponse

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def stats(store: StoreDep, pool: PoolDep) -> StatsResponse:
    counts = store.stats()
    return StatsResponse(
        pool_workers=pool.max_workers,
        jobs_total=counts.jobs_total,
        jobs_active=counts.jobs_active,
        segments_pending=counts.segments_pending,
        segments_running=counts.segments_running,
    )
