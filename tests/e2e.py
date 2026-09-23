"""
End-to-end check of the prediction service against the real library.

Boots its own uvicorn (single worker, free port), drives it over HTTP with
the stdlib only, asserts the expected outcome of each scenario, then stops
the server with SIGINT while work is in flight and checks the shutdown and
the logs. Exit code 0 only if every check passed — usable from make or CI.

    python tests/e2e.py            # full catalog, 8 happy segments (~2-3 min on 8 cores)
    python tests/e2e.py --quick    # 2 happy segments (~1-2 min)

Timings above are on AC power; fits are CPU-bound, so battery/power-saver
can make them ~3x slower. Budgets scale via E2E_SEGMENT_BUDGET_S.

Server output is kept in .e2e/server.log for inspection.
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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / ".e2e" / "server.log"
TERMINAL = {"success", "partial_success", "failed"}
STATUS_ORDER = {"pending": 0, "running": 1, "success": 2, "partial_success": 2, "failed": 2}

FULL = {"games": ["gameA", "gameB"], "regions": ["EU", "NA"], "platforms": ["Android", "iOS"]}
QUICK = {"games": ["gameA"], "regions": ["EU"], "platforms": ["Android", "iOS"]}
ALL_BAD = {"games": ["gameC"], "regions": ["EU"], "platforms": ["iOS"]}
MIXED = {"games": ["gameA", "gameC"], "regions": ["NA"], "platforms": ["iOS"]}
# ~27s nominal, but CPU-bound: on a laptop in power-saver a segment was
# measured at ~80s. Override with E2E_SEGMENT_BUDGET_S on slower machines.
SECONDS_PER_SEGMENT_BUDGET = float(os.environ.get("E2E_SEGMENT_BUDGET_S", "120"))


class Client:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool]] = []

    def check(self, name: str, ok: bool, detail: object = "") -> bool:
        self.results.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail != "" else ""), flush=True)
        return ok

    @property
    def passed(self) -> bool:
        return all(ok for _, ok in self.results)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    LOG_PATH.parent.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "pythia_service.api.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
        cwd=ROOT, env=env, stdout=LOG_PATH.open("w"), stderr=subprocess.STDOUT,
    )


def wait_ready(client: Client, server: subprocess.Popen, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited during startup; see {LOG_PATH}")
        try:
            if client.call("GET", "/health")[0] == 200:
                return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"server not ready after {timeout}s; see {LOG_PATH}")


def wait_terminal(client: Client, job_id: str, timeout: float) -> tuple[dict, list[str], set[str]]:
    """Poll until the job is terminal. Returns the final body, the sequence
    of distinct job statuses observed, and every segment stage observed."""
    statuses: list[str] = []
    stages: set[str] = set()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, body = client.call("GET", f"/predictions/{job_id}")
        if not statuses or statuses[-1] != body["status"]:
            statuses.append(body["status"])
        stages |= {s["current_stage"] for s in body["segments"] if s["current_stage"]}
        if body["status"] in TERMINAL:
            return body, statuses, stages
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} not terminal after {timeout:.0f}s")


def observed_segment_seconds(job: dict) -> float:
    """Slowest segment of a finished job, from its own timestamps."""
    spans = [
        (datetime.fromisoformat(s["finished_at"]) - datetime.fromisoformat(s["started_at"])).total_seconds()
        for s in job["segments"] if s["started_at"] and s["finished_at"]
    ]
    return max(spans, default=SECONDS_PER_SEGMENT_BUDGET)


def segment_count(payload: dict) -> int:
    return len(payload["games"]) * len(payload["regions"]) * len(payload["platforms"])


def load_expected() -> dict[tuple[str, str, str], tuple[float, float]]:
    with (ROOT / "example-results.csv").open() as f:
        return {(r["game"], r["region"], r["platform"]): (float(r["sum"]), float(r["mean"])) for r in csv.DictReader(f)}


def run(quick: bool) -> bool:
    happy = QUICK if quick else FULL
    port = free_port()
    client = Client(f"http://127.0.0.1:{port}")
    report = Report()
    server = start_server(port)
    print(f"server on :{port}, log at {LOG_PATH}")
    try:
        wait_ready(client, server)

        print("\n[readiness + contract]")
        code, health = client.call("GET", "/health")
        report.check("health: 200 with pool size", code == 200 and health["status"] == "ok", f"pool_workers={health.get('pool_workers')}")
        pool_workers = health["pool_workers"]

        invalid = {
            "empty list": {**happy, "games": []},
            "blank value": {**happy, "regions": ["  "]},
            "fan-out over the cap (75 segments)": {
                "games": [f"g{i}" for i in range(5)], "regions": [f"r{i}" for i in range(5)], "platforms": ["p1", "p2", "p3"],
            },
        }
        for name, payload in invalid.items():
            code, _ = client.call("POST", "/predictions", payload)
            report.check(f"validation: {name} -> 422", code == 422, f"got {code}")
        code, _ = client.call("GET", "/predictions/00000000-0000-0000-0000-000000000000")
        report.check("unknown job_id -> 404", code == 404, f"got {code}")

        print("\n[submission + dedup]")
        # Failure scenarios first: the pool is FIFO, so they resolve early.
        _, all_bad = client.call("POST", "/predictions", ALL_BAD)
        _, mixed = client.call("POST", "/predictions", MIXED)

        # 20 identical submissions released together; half send the lists in
        # reverse order — the dedup key must not depend on order.
        reversed_happy = {k: list(reversed(v)) for k, v in happy.items()}
        barrier = threading.Barrier(20)
        responses: list[tuple[int, dict]] = []

        def submit(i: int) -> None:
            barrier.wait()
            responses.append(client.call("POST", "/predictions", happy if i % 2 else reversed_happy))

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        job_ids = {body["job_id"] for _, body in responses}
        created = sum(not body["deduplicated"] for _, body in responses)
        report.check("dedup: 20 concurrent identical POSTs -> 1 job", len(job_ids) == 1 and created == 1,
                     f"distinct_jobs={len(job_ids)} created={created}")
        report.check("dedup: every response is 202", {c for c, _ in responses} == {202}, {c for c, _ in responses})
        happy_id = job_ids.pop()

        total_segments = segment_count(ALL_BAD) + segment_count(MIXED) + segment_count(happy)
        budget = math.ceil(total_segments / pool_workers) * SECONDS_PER_SEGMENT_BUDGET + 30
        print(f"\n[execution — {total_segments} segments on {pool_workers} slots, budget {budget}s]")

        final, _, _ = wait_terminal(client, all_bad["job_id"], budget)
        seg = final["segments"][0]
        report.check("total failure: unknown segment -> job failed", final["status"] == "failed", final["status"])
        report.check("total failure: failed_stage and readable error",
                     seg["failed_stage"] == "fetching_players" and seg["error_message"].startswith("KeyError: No data configured"),
                     seg["error_message"])

        final, _, _ = wait_terminal(client, mixed["job_id"], budget)
        report.check("partial failure: 1 good + 1 unknown -> partial_success",
                     final["status"] == "partial_success" and final["segments_succeeded"] == 1 and final["segments_failed"] == 1,
                     f"{final['status']} ok={final['segments_succeeded']} failed={final['segments_failed']}")

        final, statuses, stages = wait_terminal(client, happy_id, budget)
        report.check("happy path: job success", final["status"] == "success", final["status"])
        report.check("progress: polling exposed intermediate stages", len(stages) >= 3, sorted(stages))
        report.check("progress: job status never regressed",
                     all(STATUS_ORDER[a] <= STATUS_ORDER[b] for a, b in zip(statuses, statuses[1:])), " -> ".join(statuses))

        expected = load_expected()
        mismatches = [
            (s["game"], s["region"], s["platform"]) for s in final["segments"]
            if not (math.isclose(s["prediction"]["sum"], expected[(s["game"], s["region"], s["platform"])][0], rel_tol=1e-12)
                    and math.isclose(s["prediction"]["mean"], expected[(s["game"], s["region"], s["platform"])][1], rel_tol=1e-12))
        ]
        report.check(f"results: {len(final['segments'])} segments match example-results.csv", not mismatches, mismatches or "")

        code, again = client.call("POST", "/predictions", happy)
        report.check("dedup is not a cache: resubmit after finish -> new job",
                     code == 202 and not again["deduplicated"] and again["job_id"] != happy_id)

        print("\n[shutdown with work in flight]")
        time.sleep(3)  # let the resubmitted job occupy the pool
        # Shutdown waits out the segments already running, so the bound comes
        # from how long a segment actually took here, not from a guess: the
        # same fits run 3x slower on a laptop in power-saver.
        shutdown_budget = 2 * observed_segment_seconds(final) + 30
        t0 = time.monotonic()
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=shutdown_budget)
            elapsed = time.monotonic() - t0
            report.check("shutdown: SIGINT exits cleanly within one segment's time",
                         server.returncode == 0,
                         f"{elapsed:.1f}s of {shutdown_budget:.0f}s allowed, exit={server.returncode}")
        except subprocess.TimeoutExpired:
            report.check("shutdown: SIGINT exits cleanly within one segment's time", False, f"still running after {shutdown_budget:.0f}s")

        print("\n[logs]")
        log = LOG_PATH.read_text()
        events = ["service_started", "job_accepted", "job_deduplicated", "segment_started",
                  "segment_succeeded", "segment_failed", "job_finished", "service_stopping", "service_stopped"]
        missing = [e for e in events if f"event={e}" not in log]
        report.check("logs: every lifecycle event present", not missing, f"missing={missing}" if missing else "")
        report.check("logs: failures carry a traceback", "Traceback (most recent call last)" in log)
        # Counting lines would depend on whether the resubmitted job had time
        # to finish before shutdown. The invariant that matters is per job:
        # the three jobs driven above each closed exactly once, and no job
        # closed twice.
        closed = Counter(re.findall(r"event=job_finished job_id=(\S+)", log))
        driven = [all_bad["job_id"], mixed["job_id"], happy_id]
        report.check("logs: job_finished exactly once per job",
                     all(closed[job] == 1 for job in driven) and max(closed.values()) == 1,
                     dict(closed))
        waits = [float(w) for w in re.findall(r"event=segment_started .*?queued_s=([\d.]+)", log)]
        if waits:
            print(f"  info  max queue wait {max(waits):.1f}s over {len(waits)} segments (backpressure)")
    finally:
        if server.poll() is None:
            server.kill()

    passed = sum(ok for _, ok in report.results)
    print(f"\n{passed}/{len(report.results)} checks passed")
    return report.passed


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end check of the Pythia prediction service.")
    parser.add_argument("--quick", action="store_true", help="2 happy segments instead of the full 8-segment catalog")
    args = parser.parse_args()
    sys.exit(0 if run(args.quick) else 1)


if __name__ == "__main__":
    main()
