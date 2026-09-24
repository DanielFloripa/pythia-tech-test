"""
Segment execution, with the library replaced by stubs.

The real fits cost about 27 seconds; what matters here is what the pipeline
records around them, especially when something raises.
"""

import pytest

from pythia_service.domain.dedup import compute_dedup_key
from pythia_service.domain.models import JobStatus, PipelineStage, PredictionRequest, SegmentStatus
from pythia_service.worker import pipeline
from tests.unit.conftest import FailingExecutor

SEGMENT = ("gameA", "EU", "iOS")
OTHER_SEGMENT = ("gameA", "EU", "Android")


class FakePrediction:
    """What the pipeline actually needs from the library's ndarray."""

    def sum(self) -> float:
        return 42.0

    def mean(self) -> float:
        return 0.5


class FakeAdapter:
    """Records the calls so a test can assert the pipeline's order, and can
    be told to fail at one specific stage."""

    def __init__(self, fail_on: str | None = None, error: Exception | None = None) -> None:
        self.fail_on = fail_on
        self.error = error or KeyError("No data configured for: game='gameC'")
        self.calls: list[str] = []

    def _record(self, name: str):
        self.calls.append(name)
        if name == self.fail_on:
            raise self.error

    def fetch_players(self, *_): self._record("fetch_players"); return "players"
    def fetch_revenues(self, *_): self._record("fetch_revenues"); return "revenues"
    def fetch_gamerounds(self, *_): self._record("fetch_gamerounds"); return "gamerounds"
    def fit_players(self, *_): self._record("fit_players"); return "modeled_players"
    def fit_economy(self, *_): self._record("fit_economy"); return "modeled_economy"
    def fit_gamerounds(self, *_): self._record("fit_gamerounds"); return "modeled_gamerounds"
    def run_prediction(self, *_): self._record("run_prediction"); return FakePrediction()


@pytest.fixture
def adapter(monkeypatch):
    def install(fail_on: str | None = None, error: Exception | None = None) -> FakeAdapter:
        fake = FakeAdapter(fail_on=fail_on, error=error)
        monkeypatch.setattr(pipeline, "pythia_adapter", fake)
        return fake

    return install


@pytest.fixture
def job(store, prediction_request):
    created, _ = store.get_or_create_job(
        prediction_request, compute_dedup_key(prediction_request), [SEGMENT, OTHER_SEGMENT]
    )
    return created


def test_a_successful_segment_records_its_prediction(store, job, adapter):
    adapter()

    pipeline.process_segment(store, job.job_id, SEGMENT)

    segment = store.get_job(job.job_id).segments[SEGMENT]
    assert segment.status == SegmentStatus.SUCCESS
    assert (segment.prediction.sum, segment.prediction.mean) == (42.0, 0.5)
    assert segment.started_at is not None and segment.finished_at is not None
    assert segment.current_stage is None and segment.error_message is None


def test_the_pipeline_runs_the_stages_in_order(store, job, adapter):
    fake = adapter()

    pipeline.process_segment(store, job.job_id, SEGMENT)

    assert fake.calls == [
        "fetch_players", "fetch_revenues", "fetch_gamerounds",
        "fit_players", "fit_economy", "fit_gamerounds", "run_prediction",
    ]


def test_a_failure_is_recorded_where_it_happened_and_never_raises(store, job, adapter):
    adapter(fail_on="fit_economy", error=ValueError("boom"))

    pipeline.process_segment(store, job.job_id, SEGMENT)  # must not raise

    segment = store.get_job(job.job_id).segments[SEGMENT]
    assert segment.status == SegmentStatus.FAILED
    assert segment.failed_stage == PipelineStage.FITTING_ECONOMY
    assert segment.error_message == "ValueError: boom"
    assert segment.prediction is None


def test_a_key_error_reads_as_a_sentence(store, job, adapter):
    # str() on a KeyError wraps the message in quotes; the API should not.
    adapter(fail_on="fetch_players", error=KeyError("No data configured for: game='gameC'"))

    pipeline.process_segment(store, job.job_id, SEGMENT)

    message = store.get_job(job.job_id).segments[SEGMENT].error_message
    assert message == "KeyError: No data configured for: game='gameC'"


def test_one_failed_segment_leaves_the_others_alone(store, job, adapter):
    adapter(fail_on="fetch_players")
    pipeline.process_segment(store, job.job_id, SEGMENT)
    adapter()
    pipeline.process_segment(store, job.job_id, OTHER_SEGMENT)

    assert store.get_job(job.job_id).status == JobStatus.PARTIAL_SUCCESS


def test_submitting_hands_every_segment_to_the_pool(store, job, executor):
    pipeline.submit_job_segments(store, executor, job.job_id, [SEGMENT, OTHER_SEGMENT])

    submitted = [args[2] for _, args, _ in executor.submitted]
    assert submitted == [SEGMENT, OTHER_SEGMENT]


def test_a_dying_pool_leaves_no_segment_pending(store, job):
    """P3b: the pool shuts down mid-loop. Whatever is left pending would
    strand the job forever and keep its dedup key occupied, so every later
    identical request would join a job that can never finish. The segment
    that was already queued counts too: the pool's shutdown cancels it."""
    executor = FailingExecutor(accept=1)

    with pytest.raises(RuntimeError):
        pipeline.submit_job_segments(store, executor, job.job_id, [SEGMENT, OTHER_SEGMENT])

    finished = store.get_job(job.job_id)
    assert finished.status == JobStatus.FAILED
    assert [s.status for s in finished.segments.values()] == [SegmentStatus.FAILED] * 2
    assert all(s.error_message == "not scheduled: service shutting down"
               for s in finished.segments.values())
    assert executor.futures[0].cancelled()


def test_a_segment_already_running_is_left_alone(store, job):
    """cancel() refuses work that started, and it should: that thread is
    mid-fit and will write its own result."""
    executor = FailingExecutor(accept=1)
    original_submit = executor.submit

    def submit_then_start(*args, **kwargs):
        future = original_submit(*args, **kwargs)
        executor.start_last()
        return future

    executor.submit = submit_then_start

    with pytest.raises(RuntimeError):
        pipeline.submit_job_segments(store, executor, job.job_id, [SEGMENT, OTHER_SEGMENT])

    segments = store.get_job(job.job_id).segments
    assert segments[SEGMENT].status == SegmentStatus.PENDING  # still owned by its thread
    assert segments[OTHER_SEGMENT].status == SegmentStatus.FAILED


def test_nothing_scheduled_at_all_still_closes_the_job(store, job):
    with pytest.raises(RuntimeError):
        pipeline.submit_job_segments(store, FailingExecutor(accept=0), job.job_id, [SEGMENT, OTHER_SEGMENT])

    assert store.get_job(job.job_id).status == JobStatus.FAILED
