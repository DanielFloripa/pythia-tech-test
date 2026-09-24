"""
Fixtures shared by the unit tests.

None of these tests touch pythia_library: fits cost ~27s each and the point
here is the orchestration around them, not the modelling. The library is
replaced by stubs and the pool by executors whose behaviour each test picks.
"""

from collections.abc import Callable
from concurrent.futures import Future

import pytest
from fastapi.testclient import TestClient

from pythia_service.api.dependencies import get_pool, get_store
from pythia_service.api.main import app
from pythia_service.domain.models import PredictionRequest
from pythia_service.jobs.store import JobStore


class RecordingExecutor:
    """Accepts work and never runs it, so jobs stay pending and assertions
    stay deterministic. Hands back real Futures, because the scheduling code
    cancels them when the pool dies mid-loop."""

    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.futures: list[Future] = []

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        self.submitted.append((fn, args, kwargs))
        future: Future = Future()
        self.futures.append(future)
        return future

    def start_last(self) -> None:
        """Mark the most recent submission as already running, which is what
        makes cancel() refuse it."""
        self.futures[-1].set_running_or_notify_cancel()


class FailingExecutor(RecordingExecutor):
    """Accepts `accept` submissions, then behaves like a shut-down executor.
    This is the P3b race: the pool dies between the endpoint's check and its
    submit loop."""

    def __init__(self, accept: int = 0) -> None:
        super().__init__()
        self.accept = accept

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        if len(self.submitted) >= self.accept:
            raise RuntimeError("cannot schedule new futures after shutdown")
        return super().submit(fn, *args, **kwargs)


class FakePool:
    """Stands in for SegmentPool. `executor=None` means "already shut down",
    which is what makes /health answer 503 and POST refuse up front."""

    def __init__(self, executor: RecordingExecutor | None = None, max_workers: int = 2) -> None:
        self._executor = executor
        self.max_workers = max_workers

    @property
    def is_running(self) -> bool:
        return self._executor is not None

    @property
    def executor(self) -> RecordingExecutor:
        if self._executor is None:
            raise RuntimeError("SegmentPool.start() must be called before submitting work")
        return self._executor


@pytest.fixture
def store() -> JobStore:
    return JobStore()


@pytest.fixture
def executor() -> RecordingExecutor:
    return RecordingExecutor()


@pytest.fixture
def make_client(store: JobStore) -> Callable[..., TestClient]:
    """Builds a TestClient wired to the given pool. The app's lifespan still
    runs, so the wiring under test is the real one; only the two dependencies
    are swapped."""

    def build(pool: FakePool | None = None, job_store: JobStore | None = None) -> TestClient:
        app.dependency_overrides[get_store] = lambda: job_store or store
        app.dependency_overrides[get_pool] = lambda: pool or FakePool(RecordingExecutor())
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


@pytest.fixture
def prediction_request() -> PredictionRequest:
    return PredictionRequest(games=["gameA"], regions=["EU"], platforms=["iOS", "Android"])
