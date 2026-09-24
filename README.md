> **Submission** — the original challenge text is preserved below, unchanged.
>
> - **[SOLUTION.md](SOLUTION.md)** — how to run it, the endpoints, where each module lives
> - **[DESIGN.md](DESIGN.md)** — the one-page design doc: components, request flow, failure modes
>
> Quick start: `make install-dev && make test && make run`

---

# Pythia — Backend Engineer Tech Test

## Context

A Data Science team has built **Pythia**: a player behaviour prediction pipeline
for our mobile games. The prototype lives in `pythia_library_v1/` (the DS internals)
and `scripts/pythia-prediction-v1.py` (its public interface), and runs as a script
on a single machine. It takes several minutes to produce results for a full set
of games, regions, and platforms.

The business now needs Pythia to be available **on demand** via an HTTP API.
Multiple internal teams will call it simultaneously. Your job is to design and
build the production layer around the library.

---

## How the pipeline works

Pythia generates predictions for every combination of **game × region × platform**.
Each combination is called a **segment** — the atomic unit of work.

For each segment the pipeline:

1. Fetches three datasets from BigQuery — `players`, `revenues`, `gamerounds`
2. Fits a sequence of models on that data (`fit_player_data` → `fit_economy_data` → `fit_gameround_data`)
3. Produces a numeric prediction array (`predict`)

BigQuery queries take several seconds each (see `pythia_library_v1/bigquery.py` for exact latencies).
The model-fitting steps are CPU-intensive. Together, one segment takes roughly a minute end-to-end;
the default run covers 8 segments (2 games × 2 regions × 2 platforms) and takes several minutes.

Run `./run-v1.sh` to see this in action. `example-results.csv` shows the expected output —
one row per segment with a `sum` and a `mean` over the prediction array.

### `pythia_library_v1` — cost per call

| Call | Wall time | CPU cores | Blocks caller? |
|---|---|---|---|
| `query_data("players", ...)` | 5 s | — | Yes (I/O wait) |
| `query_data("revenues", ...)` | 3 s | — | Yes (I/O wait) |
| `query_data("gamerounds", ...)` | 2 s | — | Yes (I/O wait) |
| `fit_player_data(data)` | ~5 s | 4 | Yes (CPU) |
| `fit_economy_data(data, players)` | ~5 s | 4 | Yes (CPU) |
| `fit_gameround_data(data, economy)` | ~4 s | 4 | Yes (CPU) |
| `predict(players, economy, gamerounds)` | ~3 s | 4 | Yes (CPU) |
| **1 segment, sequential** | **~27 s** | | |

The modeling functions each spawn an internal process pool — actual wall time may be slightly
higher than the figures above on first call due to pool start-up and calibration.

---

## The library

`pythia_library_v1/` and `scripts/pythia-prediction-v1.py` together represent the DS
team's packaged code. In a real project this would be installed as
`pip install pythia` — treat it accordingly.

| File | Status | Why |
|---|---|---|
| `pythia_library_v1/bigquery.py` | **Do not modify** | Owned by Data Engineering; simulates the company's BigQuery client |
| `pythia_library_v1/modeling.py` | **Do not modify** | Owned by Data Science; the modelling framework |
| `scripts/pythia-prediction-v1.py` | **You may adapt** | The library's public interface — if you need to change the signature to suit your backend, do so and document why |

Run `./run-v1.sh` to see the library working end-to-end and compare the output
against `example-results.csv` before you start building.

**Important boundary:** the library must remain usable without any backend
infrastructure. Any infrastructure concern (concurrency control, rate limiting,
job tracking) belongs in the backend — not in the library. If you find yourself
adding infrastructure dependencies to `pythia_library_v1/` or
`scripts/pythia-prediction-v1.py`, reconsider the design.

---

## Your task

Build a backend service that exposes Pythia via HTTP.

### Functional requirements

1. **Submit a prediction request** — the caller provides a list of games,
   regions, and platforms.

2. **Track the outcome** — the caller can follow the progress of a submitted
   request.

3. **Retrieve results** — once complete, the caller can fetch the per-segment
   predictions.

4. **Concurrent requests** — at least 20 simultaneous prediction requests must
   be handled without degrading each other.

### Non-functional requirements

- A single prediction run for all 8 segments takes several minutes of real
  compute time. Design accordingly.
- The system should remain correct if a single segment's data fetch or
  modelling step fails — decide how partial failures are surfaced.
- Think about what happens when the same (games, regions, platforms) combination
  is submitted multiple times within a short window.

---

## Deliverables

**1. Design document** (max one page)

- What are the main components of your system?
- How does a request flow from submission to result retrieval?
- What failure modes are you aware of, and how does the system behave?

**2. API contract** — well-typed request and response schemas for every endpoint.

**3. FastAPI endpoints** — at minimum:
- `POST /predictions` — submit a job
- `GET  /predictions/{job_id}` — poll status / retrieve results

**4. Background execution layer** — the component that runs the Pythia pipeline
outside the HTTP request/response cycle. You do not need a fully wired-up
solution; a clean skeleton that shows your approach and its seams is enough.

---

## What we are not looking for

- A fully running, Docker-packaged system
- Cloud infrastructure or Kubernetes configuration
- Changes to `pythia_library_v1/bigquery.py` or `pythia_library_v1/modeling.py`
- Reimplementing the Pythia library from scratch

---

There is no pre-existing backend code. Start from scratch.

---

## Part 2 — Optimized pipeline

The Data Science team has updated the BigQuery client: queries now take
significantly longer. The updated library is in `pythia_library_v2/` — use it
instead of `pythia_library_v1/` for this task.

### `pythia_library_v2` — cost per call

| Call | Wall time | CPU cores | Blocks caller? |
|---|---|---|---|
| `query_data("players", ...)` | 15 s | — | Yes (I/O wait) |
| `query_data("revenues", ...)` | 10 s | — | Yes (I/O wait) |
| `query_data("gamerounds", ...)` | 8 s | — | Yes (I/O wait) |
| `fit_player_data(data)` | ~5 s | 4 | Yes (CPU) |
| `fit_economy_data(data, players)` | ~5 s | 4 | Yes (CPU) |
| `fit_gameround_data(data, economy)` | ~4 s | 4 | Yes (CPU) |
| `predict(players, economy, gamerounds)` | ~3 s | 4 | Yes (CPU) |
| **1 segment, sequential** | **~50 s** | | |

With the new latencies, the sequential fetch in `scripts/pythia-prediction-v1.py`
becomes a bottleneck. A colleague from the DS team has proposed a new design
where I/O and modelling overlap within each segment:

```
fetch players  ─────────────────┐
                                 fit_players
                 fetch revenues  ──────┐
                                        fit_economy
                          fetch gamerounds  ──┐
                                               fit_gamerounds → predict
```

A skeleton is provided in `scripts/pythia-prediction-v2-optimized.py`.
The `_process_segment` function is already there — complete the script:

1. Implement `query_data_async`: it starts the query in a background thread
   immediately (non-blocking) and returns a handle whose `["column"]` access
   blocks until the data is ready.
2. Implement `main()` using the same argument interface as
   `scripts/pythia-prediction-v1.py`, processing segments concurrently.

Validate your solution:

- `./run-v2-optimized.sh` produces `results-v2-optimized.csv`
- Values in `results-v2-optimized.csv` match `example-results.csv`
- `./run-v2-optimized.sh` is measurably faster than `./run-v2.sh` (both use `pythia_library_v2`)

---

## Glossary

| Term | Meaning |
|---|---|
| **Segment** | One (game, region, platform) combination — the atomic unit of prediction |
| **KPI** | One of the three data feeds: `players`, `revenues`, `gamerounds` |
| **BigQuery** | The company's data warehouse; the library simulates its query latency |
| **Fit** | A CPU-intensive model training step (`fit_player_data`, `fit_economy_data`, `fit_gameround_data`) |
| **Predict** | Final step that produces a numeric array of predictions for a segment |
| **Pipeline** | The full sequence for one segment: fetch KPIs → fit models → predict |
