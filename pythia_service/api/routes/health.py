"""
Readiness: 200 while the pool takes work, 503 after shutdown. Shaped for a
k8s probe or a load balancer, and used by tests/e2e.py to know when the
server is up.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from pythia_service.api.dependencies import PoolDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    pool_workers: int


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Pool is shut down"}},
)
async def health(pool: PoolDep) -> HealthResponse:
    if not pool.is_running:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "service is shutting down")
    return HealthResponse(status="ok", pool_workers=pool.max_workers)
