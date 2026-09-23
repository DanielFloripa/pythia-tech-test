"""
How handlers reach the one store and the one pool.

They are built by the lifespan and kept on app.state, so nothing exists at
import time and a test can inject its own through
app.dependency_overrides[get_store].
"""

from typing import Annotated

from fastapi import Depends, Request

from pythia_service.jobs.store import JobStore
from pythia_service.worker.pool import SegmentPool


def get_store(request: Request) -> JobStore:
    return request.app.state.store


def get_pool(request: Request) -> SegmentPool:
    return request.app.state.pool


StoreDep = Annotated[JobStore, Depends(get_store)]
PoolDep = Annotated[SegmentPool, Depends(get_pool)]
