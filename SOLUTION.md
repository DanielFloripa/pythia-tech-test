# Pythia — solution

An HTTP service that runs the Pythia pipeline as background jobs, plus the
optimized Part 2 script. The reasoning behind every decision is in
[DESIGN.md](DESIGN.md); this page is how to run it and where things live.

## Run it

```bash
make install-dev          # venv (Python 3.14) + runtime and test deps
make test                 # 56 unit tests, ~0.1s, the library is never called
make run                  # API on :8000, single worker
```

Then, in another shell:

```bash
curl -s -X POST localhost:8000/predictions \
  -H 'content-type: application/json' \
  -d '{"games":["gameA"],"regions":["EU"],"platforms":["iOS","Android"]}'
# {"job_id":"…","status":"pending","deduplicated":false,"segment_count":2,…}

curl -s localhost:8000/predictions/<job_id>   # per-segment stage, result or error
curl -s localhost:8000/predictions            # newest jobs, summaries only
curl -s localhost:8000/stats                  # pool size, pending and running segments
```

A segment takes about 27s and a full run is 8 of them, so expect minutes, not
seconds. `make run-debug` adds one log line per stage transition.

| Endpoint | Purpose |
|---|---|
| `POST /predictions` | Submit. Always 202: work is asynchronous either way. Returns `deduplicated: true` when an identical request is already in flight. 422 if the request expands past `PYTHIA_MAX_SEGMENTS`. |
| `GET /predictions/{job_id}` | Job status plus every segment: its stage while running, its `sum`/`mean` when done, its `failed_stage` and message when not. 404 if unknown. |
| `GET /predictions` | Summaries, newest first. `limit` (1–100, default 20) and optional `status`. |
| `GET /stats` | Queue depth: `segments_pending`, `segments_running`, `jobs_active`, `pool_workers`. |
| `GET /health` | Readiness. 503 once the pool has been shut down. |

Full schemas are at `/docs` while the service is running.

## Where things are

```
pythia_service/
├── api/
│   ├── main.py              app + lifespan: creates the store and the pool, shuts them down
│   ├── dependencies.py      how handlers reach those two singletons
│   └── routes/              predictions.py (submit, poll, list), stats.py, health.py
├── domain/
│   ├── models.py            the wire contract, plus the fan-out cap
│   └── dedup.py             identity of a submission
├── jobs/
│   ├── store.py             job and segment state, dedup index, the lock
│   └── state.py             segment statuses -> one job status (pure function)
└── worker/
    ├── pool.py              the bounded thread pool segments run on
    ├── pipeline.py          one segment, start to finish, and its logging
    └── pythia_adapter.py    the only module that imports pythia_library

tests/
├── unit/                    fast: fake adapter, fake pool, TestClient
└── e2e.py                   boots a real server and drives it over HTTP

scripts/pythia-prediction-v2-optimized.py   Part 2
```

## Verify it

```bash
make test        # unit: contract, store invariants, pipeline, endpoints
make e2e         # end to end against a real server (~3 min, or ~7 on battery)
make check       # both
```

`make e2e` starts its own uvicorn on a free port, drives 11 segments through
it, interrupts it mid-flight to check the shutdown, and reads the logs back.
Its output lands in `.e2e/server.log`. `make e2e-quick` is the shorter variant.

Part 2, measured on the same machine and power state:

```bash
make run-v2             # baseline: 905s
make run-v2-optimized   # 182s, same values as example-results.csv
```

## Knobs

| Variable | Default | What it changes |
|---|---|---|
| `PYTHIA_MAX_SEGMENTS` | 64 | Largest fan-out a single request may ask for. |
| `LOG_LEVEL` | `INFO` | `DEBUG` adds a line per pipeline stage. |
| `PORT` | 8000 | `make run` only. |
| `E2E_SEGMENT_BUDGET_S` | 120 | How long the end-to-end run allows per segment. |

## Two things worth knowing before running it

**One worker.** Job state lives in this process's memory, so a second uvicorn
worker would keep its own jobs and answer 404 for the first one's ids. The
`Makefile` always passes `--workers 1`.

**Timings move with CPU power state.** The fits are CPU-bound. The same
segment measured ~27s on AC and ~80s on a laptop in power-saver, so the
numbers above assume the former.

[DESIGN.md](DESIGN.md) covers the components, the failure modes and the
limits this design accepts.
