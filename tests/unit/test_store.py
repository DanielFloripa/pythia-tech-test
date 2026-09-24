"""
JobStore invariants. This is where the concurrency bugs live, so most of
these assert something that only breaks under interleaving.
"""

import threading

from pythia_service.domain.dedup import compute_dedup_key
from pythia_service.domain.models import JobStatus, PredictionRequest, SegmentResult, SegmentStatus
from pythia_service.jobs.store import JobStore

SEGMENTS = [("gameA", "EU", "iOS"), ("gameA", "EU", "Android")]


def _terminal(key, status=SegmentStatus.SUCCESS) -> SegmentResult:
    game, region, platform = key
    return SegmentResult(game=game, region=region, platform=platform, status=status)


def _create(store: JobStore, request: PredictionRequest, segments=None):
    segments = segments or request.segment_keys()
    return store.get_or_create_job(request, compute_dedup_key(request), segments)


def test_first_submission_creates_and_the_next_one_joins_it(store, prediction_request):
    job, created = _create(store, prediction_request)
    again, created_again = _create(store, prediction_request)

    assert created is True
    assert created_again is False
    assert again.job_id == job.job_id


def test_concurrent_identical_submissions_produce_one_job(store, prediction_request):
    # The TOCTOU regression test: with check and insert as separate calls, the
    # lock is released in between and several of these threads create a job.
    thread_count = 50
    barrier = threading.Barrier(thread_count)
    outcomes: list[tuple] = []

    def submit(_: int) -> None:
        barrier.wait()
        outcomes.append(_create(store, prediction_request))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({job.job_id for job, _ in outcomes}) == 1
    assert sum(created for _, created in outcomes) == 1


def test_a_snapshot_does_not_change_underneath_its_reader(store, prediction_request):
    job, _ = _create(store, prediction_request, SEGMENTS)
    snapshot = store.get_job(job.job_id)

    store.update_segment(job.job_id, SEGMENTS[0], _terminal(SEGMENTS[0]))

    assert snapshot.segments[SEGMENTS[0]].status == SegmentStatus.PENDING
    assert store.get_job(job.job_id).segments[SEGMENTS[0]].status == SegmentStatus.SUCCESS


def test_status_and_segments_in_one_snapshot_always_agree(store, prediction_request):
    job, _ = _create(store, prediction_request, SEGMENTS)
    for key in SEGMENTS:
        store.update_segment(job.job_id, key, _terminal(key))

    snapshot = store.get_job(job.job_id)
    finished = all(s.status == SegmentStatus.SUCCESS for s in snapshot.segments.values())
    assert snapshot.status == JobStatus.SUCCESS
    assert finished


def test_finishing_a_job_frees_its_dedup_key(store, prediction_request):
    job, _ = _create(store, prediction_request, SEGMENTS)
    for key in SEGMENTS:
        store.update_segment(job.job_id, key, _terminal(key))

    fresh, created = _create(store, prediction_request, SEGMENTS)
    assert created is True
    assert fresh.job_id != job.job_id  # dedup is a short window, not a cache


def test_only_the_write_that_finishes_a_job_reports_it(store, prediction_request):
    # What keeps job_finished out of the log twice.
    job, _ = _create(store, prediction_request, SEGMENTS)

    assert store.update_segment(job.job_id, SEGMENTS[0], _terminal(SEGMENTS[0])) is False
    assert store.update_segment(job.job_id, SEGMENTS[1], _terminal(SEGMENTS[1])) is True
    assert store.update_segment(job.job_id, SEGMENTS[1], _terminal(SEGMENTS[1])) is False


def test_updating_an_unknown_job_is_a_no_op(store):
    from uuid import uuid4

    assert store.update_segment(uuid4(), SEGMENTS[0], _terminal(SEGMENTS[0])) is False


def test_unknown_job_reads_as_missing(store):
    from uuid import uuid4

    assert store.get_job(uuid4()) is None


def test_listing_is_newest_first_and_respects_the_limit(store):
    created = [
        _create(store, PredictionRequest(games=[game], regions=["EU"], platforms=["iOS"]))[0]
        for game in ("gameA", "gameB", "gameC")
    ]

    listed = store.list_jobs(limit=2)
    assert [job.job_id for job in listed] == [created[2].job_id, created[1].job_id]


def test_listing_can_filter_by_status(store, prediction_request):
    running, _ = _create(store, prediction_request, SEGMENTS)
    store.update_segment(running.job_id, SEGMENTS[0], _terminal(SEGMENTS[0]))
    other = PredictionRequest(games=["gameB"], regions=["NA"], platforms=["iOS"])
    _create(store, other)

    assert [job.job_id for job in store.list_jobs(limit=10, status=JobStatus.RUNNING)] == [running.job_id]


def test_stats_count_what_is_waiting_and_what_is_working(store, prediction_request):
    job, _ = _create(store, prediction_request, SEGMENTS)
    store.update_segment(
        job.job_id, SEGMENTS[0], _terminal(SEGMENTS[0], status=SegmentStatus.RUNNING)
    )

    stats = store.stats()
    assert (stats.jobs_total, stats.jobs_active) == (1, 1)
    assert (stats.segments_running, stats.segments_pending) == (1, 1)


def test_stats_forget_nothing_but_count_no_work_once_jobs_end(store, prediction_request):
    job, _ = _create(store, prediction_request, SEGMENTS)
    for key in SEGMENTS:
        store.update_segment(job.job_id, key, _terminal(key))

    stats = store.stats()
    assert stats.jobs_total == 1  # the store keeps finished jobs readable
    assert (stats.jobs_active, stats.segments_pending, stats.segments_running) == (0, 0, 0)
