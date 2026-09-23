"""
App wiring and process lifecycle.

Single worker, and that is a constraint, not a default: job state lives in
this process's memory, so a second worker would keep its own jobs, answer
404 for the first one's ids, and dedup nothing. PYTHONUNBUFFERED keeps the
library's own print() in step with our logs when stdout is a file.

    PYTHONUNBUFFERED=1 python -m uvicorn pythia_service.api.main:app --workers 1
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from pythia_service.api.routes.health import router as health_router
from pythia_service.api.routes.predictions import router as predictions_router
from pythia_service.domain.models import MAX_SEGMENTS_PER_REQUEST
from pythia_service.jobs.store import JobStore
from pythia_service.logging_setup import configure_logging, get_logger, kv
from pythia_service.worker.pool import SegmentPool

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    pool = SegmentPool()
    pool.start()
    app.state.store = JobStore()
    app.state.pool = pool
    logger.info(kv(
        "service_started", pool_workers=pool.max_workers, cores=os.cpu_count(),
        max_segments_per_request=MAX_SEGMENTS_PER_REQUEST,
    ))
    try:
        yield
    finally:
        # Queued segments are dropped: their results would die with this
        # process anyway. Running ones are waited out, otherwise the
        # library's child processes get torn down mid-fit.
        #
        # uvicorn only reaches this after it has closed the listener and
        # drained in-flight requests, so nothing can submit while it runs.
        # A second Ctrl+C skips it altogether.
        logger.info(kv("service_stopping", cancel_queued=True))
        pool.shutdown(wait=True, cancel_futures=True)
        logger.info(kv("service_stopped"))


app = FastAPI(title="Pythia Prediction Service", lifespan=lifespan)
app.include_router(health_router)
app.include_router(predictions_router)
