"""
End-to-end check of the prediction service against the real library.

Boots its own uvicorn (single worker, free port), drives it over HTTP with the
stdlib only, asserts the expected outcome of each scenario, then stops the
server with SIGINT while work is in flight and checks the shutdown and the
logs. Exit code 0 only if every check passed, so make and CI can use it.

    python tests/e2e.py            # full catalog, 8 happy segments
    python tests/e2e.py --quick    # 2 happy segments

Fits are CPU-bound, so a laptop on battery runs them ~3x slower than on AC.
Time budgets scale with E2E_SEGMENT_BUDGET_S; the shutdown budget is derived
from how long segments actually took in this run.

Server output lands in .e2e/server.log.
"""

import argparse
import csv
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / ".e2e" / "server.log"
EXPECTED_RESULTS = ROOT / "example-results.csv"

# Requests the scenarios drive. FULL and QUICK differ only in how much compute
# the happy path costs.
FULL_CATALOG = {"games": ["gameA", "gameB"], "regions": ["EU", "NA"], "platforms": ["Android", "iOS"]}
QUICK_CATALOG = {"games": ["gameA"], "regions": ["EU"], "platforms": ["Android", "iOS"]}
UNKNOWN_SEGMENT = {"games": ["gameC"], "regions": ["EU"], "platforms": ["iOS"]}
ONE_GOOD_ONE_UNKNOWN = {"games": ["gameA", "gameC"], "regions": ["NA"], "platforms": ["iOS"]}

TERMINAL_STATUSES = {"success", "partial_success", "failed"}
# A job may climb this scale, never fall back down it.
STATUS_RANK = {"pending": 0, "running": 1, "success": 2, "partial_success": 2, "failed": 2}
LIFECYCLE_EVENTS = (
    "service_started", "job_accepted", "job_deduplicated", "segment_started",
    "segment_succeeded", "segment_failed", "job_finished", "service_stopping", "service_stopped",
)

CONCURRENT_SUBMISSIONS = 20
READY_TIMEOUT_S = 30
POLL_INTERVAL_S = 0.5
# ~27s nominal, but a segment measured ~80s on a laptop in power-saver.
SEGMENT_BUDGET_S = float(os.environ.get("E2E_SEGMENT_BUDGET_S", "120"))


class Response(NamedTuple):
    status: int
    body: dict


class Client:
    """Returns error responses instead of raising, so asserting on a 404 or a
    422 reads the same as asserting on a 200."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def get(self, path: str) -> Response:
        return self._call("GET", path)

    def post(self, path: str, body: dict) -> Response:
        return self._call("POST", path, body)

    def _call(self, method: str, path: str, body: dict | None = None) -> Response:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return Response(response.status, json.loads(response.read()))
        except urllib.error.HTTPError as err:
            return Response(err.code, json.loads(err.read()))


class Report:
    """Prints each result as it happens, so a run that takes minutes shows
    progress instead of going silent."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool]] = []

    def section(self, title: str) -> None:
        print(f"\n[{title}]", flush=True)

    def check(self, name: str, ok: bool, detail: object = "") -> bool:
        self.results.append((name, ok))
        suffix = f"  ({detail})" if detail != "" else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}", flush=True)
        return ok

    def info(self, message: str) -> None:
        print(f"  info  {message}", flush=True)

    def summarize(self) -> bool:
        passed = sum(ok for _, ok in self.results)
        print(f"\n{passed}/{len(self.results)} checks passed")
        return passed == len(self.results)


class Server:
    """The service under test as a subprocess. A context manager, so the
    process is gone even when a scenario raises."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._process: subprocess.Popen | None = None
        self._log_file = None

    def __enter__(self) -> Server:
        LOG_PATH.parent.mkdir(exist_ok=True)
        self._log_file = LOG_PATH.open("w")
        self._process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "pythia_service.api.main:app",
             "--host", "127.0.0.1", "--port", str(self.port), "--workers", "1"],
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)},
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        if self._log_file is not None:
            self._log_file.close()

    @property
    def process(self) -> subprocess.Popen:
        if self._process is None:
            raise RuntimeError("server was not started")
        return self._process

    def wait_until_ready(self, client: Client, timeout: float = READY_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited during startup; see {LOG_PATH}")
            try:
                if client.get("/health").status == 200:
                    return
            except (urllib.error.URLError, ConnectionError):
                pass  # not listening yet
            time.sleep(0.2)
        raise TimeoutError(f"server not ready after {timeout}s; see {LOG_PATH}")

    def interrupt(self, timeout: float) -> float | None:
        """SIGINT and wait. Seconds taken, or None if it outlived the budget."""
        started = time.monotonic()
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return time.monotonic() - started

    def ensure_alive(self) -> None:
        """Fail loudly if the server died under us. Without this, the next
        request surfaces as a bare "connection refused" traceback that says
        nothing about what actually happened."""
        if self.process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {self.process.returncode} mid-run; see {LOG_PATH}"
            )

    def log(self) -> str:
        if self._log_file is not None:
            self._log_file.flush()
        return LOG_PATH.read_text()


@dataclass
class Suite:
    """What the scenarios share: the connection, the report, and the few facts
    each scenario discovers for the next one."""

    client: Client
    report: Report
    server: Server
    happy_request: dict
    pool_workers: int = 0
    job_ids: dict[str, str] = field(default_factory=dict)
    happy_job: dict = field(default_factory=dict)

    def poll_until_terminal(self, job_id: str, timeout: float) -> tuple[dict, list[str], set[str]]:
        """Poll one job to completion. Returns the final body, the distinct
        statuses seen in order, and every stage any segment passed through."""
        statuses: list[str] = []
        stages: set[str] = set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.server.ensure_alive()
            body = self.client.get(f"/predictions/{job_id}").body
            if not statuses or statuses[-1] != body["status"]:
                statuses.append(body["status"])
            stages |= {s["current_stage"] for s in body["segments"] if s["current_stage"]}
            if body["status"] in TERMINAL_STATUSES:
                return body, statuses, stages
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError(f"job {job_id} not terminal after {timeout:.0f}s")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _segment_count(request: dict) -> int:
    return len(request["games"]) * len(request["regions"]) * len(request["platforms"])


def _expected_results() -> dict[tuple[str, str, str], tuple[float, float]]:
    with EXPECTED_RESULTS.open() as handle:
        return {
            (row["game"], row["region"], row["platform"]): (float(row["sum"]), float(row["mean"]))
            for row in csv.DictReader(handle)
        }


def _mismatched_segments(segments: list[dict]) -> list[tuple[str, str, str]]:
    expected = _expected_results()
    mismatched = []
    for segment in segments:
        key = (segment["game"], segment["region"], segment["platform"])
        wanted_sum, wanted_mean = expected[key]
        got = segment["prediction"]
        if not (math.isclose(got["sum"], wanted_sum, rel_tol=1e-12)
                and math.isclose(got["mean"], wanted_mean, rel_tol=1e-12)):
            mismatched.append(key)
    return mismatched


def _slowest_segment_seconds(job: dict) -> float:
    """How long the slowest segment of a finished job took, from its own
    timestamps. The shutdown budget is built on this instead of a constant,
    because the same fits run three times slower on battery."""
    spans = [
        (datetime.fromisoformat(s["finished_at"]) - datetime.fromisoformat(s["started_at"])).total_seconds()
        for s in job["segments"] if s["started_at"] and s["finished_at"]
    ]
    return max(spans, default=SEGMENT_BUDGET_S)


def _run_in_threads(target: Callable[[int], None], count: int) -> None:
    threads = [threading.Thread(target=target, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _submit_together(client: Client, request: dict, count: int) -> list[Response]:
    """Release `count` submissions at the same instant. Half send the lists
    reversed, because the dedup key must not depend on order."""
    reversed_request = {key: list(reversed(value)) for key, value in request.items()}
    barrier = threading.Barrier(count)
    responses: list[Response] = []

    def submit(index: int) -> None:
        barrier.wait()
        responses.append(client.post("/predictions", request if index % 2 else reversed_request))

    _run_in_threads(submit, count)
    return responses


def check_readiness(suite: Suite) -> None:
    suite.report.section("readiness + contract")
    status, health = suite.client.get("/health")
    suite.pool_workers = health.get("pool_workers", 0)
    suite.report.check("health: 200 with pool size",
                       status == 200 and health["status"] == "ok",
                       f"pool_workers={suite.pool_workers}")

    rejected = {
        "empty list": {**suite.happy_request, "games": []},
        "blank value": {**suite.happy_request, "regions": ["  "]},
        "fan-out over the cap (75 segments)": {
            "games": [f"g{i}" for i in range(5)],
            "regions": [f"r{i}" for i in range(5)],
            "platforms": ["p1", "p2", "p3"],
        },
    }
    for name, request in rejected.items():
        status, _ = suite.client.post("/predictions", request)
        suite.report.check(f"validation: {name} -> 422", status == 422, f"got {status}")

    status, _ = suite.client.get("/predictions/00000000-0000-0000-0000-000000000000")
    suite.report.check("unknown job_id -> 404", status == 404, f"got {status}")


def check_submission_and_dedup(suite: Suite) -> None:
    suite.report.section("submission + dedup")
    # Failure scenarios go in first: the pool is FIFO, so they resolve early.
    suite.job_ids["unknown"] = suite.client.post("/predictions", UNKNOWN_SEGMENT).body["job_id"]
    suite.job_ids["mixed"] = suite.client.post("/predictions", ONE_GOOD_ONE_UNKNOWN).body["job_id"]

    responses = _submit_together(suite.client, suite.happy_request, CONCURRENT_SUBMISSIONS)
    job_ids = {response.body["job_id"] for response in responses}
    created = sum(not response.body["deduplicated"] for response in responses)
    suite.report.check(f"dedup: {CONCURRENT_SUBMISSIONS} concurrent identical POSTs -> 1 job",
                       len(job_ids) == 1 and created == 1,
                       f"distinct_jobs={len(job_ids)} created={created}")
    suite.report.check("dedup: every response is 202",
                       {response.status for response in responses} == {202},
                       {response.status for response in responses})
    suite.job_ids["happy"] = job_ids.pop()


def check_queue_visibility(suite: Suite) -> None:
    suite.report.section("queue visibility")
    status, stats = suite.client.get("/stats")
    in_flight = stats.get("segments_pending", 0) + stats.get("segments_running", 0)
    suite.report.check("stats: counts work in flight right after submitting",
                       status == 200 and stats["jobs_total"] == 3
                       and in_flight > 0 and stats["pool_workers"] == suite.pool_workers,
                       f"jobs={stats.get('jobs_total')} pending={stats.get('segments_pending')}"
                       f" running={stats.get('segments_running')}")

    status, listing = suite.client.get("/predictions?limit=20")
    listed_ids = [job["job_id"] for job in listing["jobs"]]
    newest_first = [job["job_id"] for job in sorted(listing["jobs"], key=lambda job: job["created_at"], reverse=True)]
    suite.report.check("list: returns the three submitted jobs, newest first",
                       status == 200 and set(listed_ids) == set(suite.job_ids.values())
                       and listed_ids == newest_first,
                       f"returned={listing.get('returned')}")
    suite.report.check("list: summaries carry no segments",
                       all("segments" not in job for job in listing["jobs"]))
    status, _ = suite.client.get("/predictions?limit=0")
    suite.report.check("list: limit=0 -> 422", status == 422, f"got {status}")


def check_execution(suite: Suite) -> None:
    total_segments = (_segment_count(UNKNOWN_SEGMENT) + _segment_count(ONE_GOOD_ONE_UNKNOWN)
                      + _segment_count(suite.happy_request))
    budget = math.ceil(total_segments / suite.pool_workers) * SEGMENT_BUDGET_S + 30
    suite.report.section(
        f"execution - {total_segments} segments on {suite.pool_workers} slots, budget {budget:.0f}s")

    failed, _, _ = suite.poll_until_terminal(suite.job_ids["unknown"], budget)
    segment = failed["segments"][0]
    suite.report.check("total failure: unknown segment -> job failed", failed["status"] == "failed", failed["status"])
    suite.report.check("total failure: failed_stage and readable error",
                       segment["failed_stage"] == "fetching_players"
                       and segment["error_message"].startswith("KeyError: No data configured"),
                       segment["error_message"])

    mixed, _, _ = suite.poll_until_terminal(suite.job_ids["mixed"], budget)
    suite.report.check("partial failure: 1 good + 1 unknown -> partial_success",
                       mixed["status"] == "partial_success"
                       and mixed["segments_succeeded"] == 1 and mixed["segments_failed"] == 1,
                       f"{mixed['status']} ok={mixed['segments_succeeded']} failed={mixed['segments_failed']}")

    happy, statuses, stages = suite.poll_until_terminal(suite.job_ids["happy"], budget)
    suite.happy_job = happy
    suite.report.check("happy path: job success", happy["status"] == "success", happy["status"])
    suite.report.check("progress: polling exposed intermediate stages", len(stages) >= 3, sorted(stages))
    suite.report.check("progress: job status never regressed",
                       all(STATUS_RANK[before] <= STATUS_RANK[after]
                           for before, after in pairwise(statuses)),
                       " -> ".join(statuses))

    mismatched = _mismatched_segments(happy["segments"])
    suite.report.check(f"results: {len(happy['segments'])} segments match example-results.csv",
                       not mismatched, mismatched or "")


def check_terminal_state(suite: Suite) -> None:
    suite.report.section("state once everything finished")
    status, filtered = suite.client.get("/predictions?status=failed")
    suite.report.check("list: status filter returns only the failed job",
                       status == 200
                       and [job["job_id"] for job in filtered["jobs"]] == [suite.job_ids["unknown"]],
                       [job["status"] for job in filtered["jobs"]])

    _, stats = suite.client.get("/stats")
    suite.report.check("stats: nothing in flight once every job is terminal",
                       stats["segments_pending"] == 0 and stats["segments_running"] == 0
                       and stats["jobs_active"] == 0,
                       f"pending={stats['segments_pending']} running={stats['segments_running']}"
                       f" active={stats['jobs_active']}")

    status, resubmitted = suite.client.post("/predictions", suite.happy_request)
    suite.report.check("dedup is not a cache: resubmit after finish -> new job",
                       status == 202 and not resubmitted["deduplicated"]
                       and resubmitted["job_id"] != suite.job_ids["happy"])


def check_shutdown(suite: Suite) -> None:
    suite.report.section("shutdown with work in flight")
    time.sleep(3)  # let the resubmitted job take over the pool
    # Shutdown waits out whatever is already running, so the bound follows what
    # a segment cost in this run rather than a guess.
    budget = 2 * _slowest_segment_seconds(suite.happy_job) + 30
    elapsed = suite.server.interrupt(timeout=budget)
    if elapsed is None:
        suite.report.check("shutdown: SIGINT exits cleanly while segments run", False,
                           f"still running after {budget:.0f}s")
        return
    suite.report.check("shutdown: SIGINT exits cleanly while segments run",
                       suite.server.process.returncode == 0,
                       f"{elapsed:.1f}s of {budget:.0f}s allowed, exit={suite.server.process.returncode}")


def check_logs(suite: Suite) -> None:
    suite.report.section("logs")
    log = suite.server.log()

    missing = [event for event in LIFECYCLE_EVENTS if f"event={event}" not in log]
    suite.report.check("logs: every lifecycle event present", not missing, f"missing={missing}" if missing else "")
    suite.report.check("logs: failures carry a traceback", "Traceback (most recent call last)" in log)

    # Counting job_finished lines would depend on whether the resubmitted job
    # finished before shutdown. The invariant that holds either way is per
    # job: each one closes exactly once.
    closed = Counter(re.findall(r"event=job_finished job_id=(\S+)", log))
    suite.report.check("logs: job_finished exactly once per job",
                       all(closed[job_id] == 1 for job_id in suite.job_ids.values())
                       and max(closed.values(), default=0) <= 1,
                       dict(closed))

    queue_waits = [float(wait) for wait in re.findall(r"event=segment_started .*?queued_s=([\d.]+)", log)]
    if queue_waits:
        suite.report.info(
            f"max queue wait {max(queue_waits):.1f}s over {len(queue_waits)} segments (backpressure)")


SCENARIOS = (
    check_readiness,
    check_submission_and_dedup,
    check_queue_visibility,
    check_execution,
    check_terminal_state,
    check_shutdown,
    check_logs,
)


def run(quick: bool) -> bool:
    report = Report()
    with Server() as server:
        client = Client(server.url)
        print(f"server on :{server.port}, log at {LOG_PATH}")
        server.wait_until_ready(client)
        suite = Suite(
            client=client,
            report=report,
            server=server,
            happy_request=QUICK_CATALOG if quick else FULL_CATALOG,
        )
        for scenario in SCENARIOS:
            try:
                scenario(suite)
            except Exception as exc:  # noqa: BLE001 - one bad scenario shouldn't hide the rest
                report.check(f"{scenario.__name__} raised {type(exc).__name__}", False, exc)
                break
    return report.summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end check of the Pythia prediction service.")
    parser.add_argument("--quick", action="store_true",
                        help="2 happy segments instead of the full 8-segment catalog")
    args = parser.parse_args()
    sys.exit(0 if run(args.quick) else 1)


if __name__ == "__main__":
    main()
