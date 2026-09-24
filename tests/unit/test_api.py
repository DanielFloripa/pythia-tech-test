"""
The HTTP surface, driven through TestClient with a fake pool.

Nothing here runs a segment: the executor only records what it was given, so
jobs stay pending and every assertion is deterministic.
"""

from tests.unit.conftest import FailingExecutor, FakePool, RecordingExecutor

CATALOG = {"games": ["gameA"], "regions": ["EU"], "platforms": ["iOS", "Android"]}
REVERSED_CATALOG = {"games": ["gameA"], "regions": ["EU"], "platforms": ["Android", "iOS"]}


def test_submitting_accepts_and_queues_every_segment(make_client, executor):
    client = make_client(FakePool(executor))

    response = client.post("/predictions", json=CATALOG)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"  # snapshot from creation, before any work
    assert body["deduplicated"] is False
    assert body["segment_count"] == 2
    assert len(executor.submitted) == 2


def test_an_identical_submission_joins_the_running_job(make_client, executor):
    client = make_client(FakePool(executor))

    first = client.post("/predictions", json=CATALOG).json()
    second = client.post("/predictions", json=REVERSED_CATALOG).json()

    assert second["job_id"] == first["job_id"]
    assert second["deduplicated"] is True
    # Scheduling again would compute everything twice and let two threads
    # write the same segment keys.
    assert len(executor.submitted) == 2


def test_polling_returns_the_segments_of_a_job(make_client):
    client = make_client()

    job_id = client.post("/predictions", json=CATALOG).json()["job_id"]
    body = client.get(f"/predictions/{job_id}").json()

    assert body["segment_count"] == 2
    assert {(s["game"], s["region"], s["platform"]) for s in body["segments"]} == {
        ("gameA", "EU", "iOS"),
        ("gameA", "EU", "Android"),
    }
    assert all(s["status"] == "pending" for s in body["segments"])


def test_an_unknown_job_is_a_404(make_client):
    client = make_client()

    response = client.get("/predictions/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404


def test_a_malformed_job_id_is_a_422(make_client):
    client = make_client()

    assert client.get("/predictions/not-a-uuid").status_code == 422


def test_an_oversized_request_is_refused_before_any_job_exists(make_client, executor):
    client = make_client(FakePool(executor))

    response = client.post("/predictions", json={
        "games": [f"g{i}" for i in range(5)],
        "regions": [f"r{i}" for i in range(5)],
        "platforms": ["p1", "p2", "p3"],
    })

    assert response.status_code == 422
    assert executor.submitted == []
    assert client.get("/stats").json()["jobs_total"] == 0


def test_listing_summarizes_jobs_newest_first(make_client):
    client = make_client()

    first = client.post("/predictions", json=CATALOG).json()["job_id"]
    second = client.post("/predictions", json={**CATALOG, "games": ["gameB"]}).json()["job_id"]

    body = client.get("/predictions?limit=10").json()
    assert [job["job_id"] for job in body["jobs"]] == [second, first]
    assert body["returned"] == 2
    assert "segments" not in body["jobs"][0]  # summaries stay small


def test_listing_requires_a_sane_limit(make_client):
    client = make_client()

    assert client.get("/predictions?limit=0").status_code == 422
    assert client.get("/predictions?limit=101").status_code == 422


def test_listing_can_filter_by_status(make_client):
    client = make_client()
    client.post("/predictions", json=CATALOG)

    assert client.get("/predictions?status=pending").json()["returned"] == 1
    assert client.get("/predictions?status=success").json()["returned"] == 0


def test_stats_expose_the_queue(make_client, executor):
    client = make_client(FakePool(executor, max_workers=3))
    client.post("/predictions", json=CATALOG)

    stats = client.get("/stats").json()

    assert stats["pool_workers"] == 3
    assert stats["jobs_total"] == 1
    assert stats["jobs_active"] == 1
    assert stats["segments_pending"] == 2
    assert stats["segments_running"] == 0


def test_health_reports_the_pool(make_client):
    client = make_client(FakePool(RecordingExecutor(), max_workers=2))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "pool_workers": 2}


def test_health_turns_503_once_the_pool_is_gone(make_client):
    client = make_client(FakePool(executor=None))

    assert client.get("/health").status_code == 503


def test_submitting_to_a_stopped_service_creates_nothing(make_client):
    client = make_client(FakePool(executor=None))

    response = client.post("/predictions", json=CATALOG)

    assert response.status_code == 503
    assert client.get("/stats").json()["jobs_total"] == 0


def test_a_job_that_cannot_be_scheduled_is_still_answerable(make_client, store):
    """The P3b path, which the end-to-end run cannot reach: uvicorn drains
    in-flight requests before shutting the pool down, so this race only
    happens under another server, or a pool stopped by other means."""
    client = make_client(FakePool(FailingExecutor(accept=1)), job_store=store)

    response = client.post("/predictions", json=CATALOG)
    assert response.status_code == 503

    listed = client.get("/predictions?limit=10").json()["jobs"][0]
    assert listed["status"] in {"failed", "running"}
    assert listed["segments_failed"] >= 1

    # And the dedup key was released, so a later identical request is not
    # merged into the job that could never run.
    working = make_client(FakePool(RecordingExecutor()), job_store=store)
    retried = working.post("/predictions", json=CATALOG).json()
    assert retried["deduplicated"] is False
    assert retried["job_id"] != listed["job_id"]
