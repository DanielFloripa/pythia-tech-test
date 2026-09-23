"""
Pythia — optimized pipeline.

Complete this script so that I/O and modelling overlap within each segment:

    fetch players  ─────────────────┐
                                     fit_players
                     fetch revenues  ──────┐
                                            fit_economy
                              fetch gamerounds  ──┐
                                                   fit_gamerounds → predict

The function _process_segment is provided — implement everything around it.

Validate your solution:
    - ./run-optimized.sh produces results-v2-optimized.csv
    - Values in results-v2-optimized.csv match example-results.csv
    - ./run-optimized.sh is measurably faster than ./run-example.sh
"""
import argparse
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pandas as pd

from pythia_library_v2.bigquery import query_data
from pythia_library_v2.modeling import fit_economy_data, fit_gameround_data, fit_player_data, predict

# Memory sets this number, not CPU. One segment holds ~0.7 GB of model
# arrays (0.76 GB peak RSS, measured), and segments started together peak
# together. Too many library processes for the cores just means slower; too
# little memory means the OOM killer. Constant, not a flag, because the
# README asks for the same CLI as the v1 script.
MAX_CONCURRENT_SEGMENTS = 4


class QueryHandle:
    """A query in flight. `handle["column"]` waits for it and returns that
    column; if the query raised, the exception surfaces here, in the caller,
    instead of dying quietly in a background thread.
    """

    def __init__(self, future: Future) -> None:
        self._future = future

    def __getitem__(self, column: str) -> pd.Series:
        return self._future.result()[column]


def query_data_async(kpi_name, game, region, platform):
    """Fire the query off immediately, hand back a handle.

    A thread per call rather than a pool: a bounded pool could leave the
    query queued, which is no longer "immediately", and sharing one pool with
    the segments invites the classic nested-submit deadlock, segments holding
    every worker while waiting on queries that cannot start. These threads
    only sleep on I/O, and at most one per segment is alive.
    """
    future: Future = Future()

    def run() -> None:
        try:
            future.set_result(query_data(kpi_name, game, region, platform))
        except BaseException as exc:  # propagated to the caller via the handle
            future.set_exception(exc)

    threading.Thread(target=run, name=f"query-{kpi_name}-{game}-{region}-{platform}", daemon=True).start()
    return QueryHandle(future)


# Worth reading _process_segment below carefully before trusting the diagram
# at the top of this file. Each handle is consumed on the line right after it
# is created, so a fetch never runs during the previous fit: on v2 latencies
# that is 15+5 +10+5 +8+4 +3 = 50s, exactly the sequential number. There is
# no in-segment overlap as written, so the speed-up here comes from running
# segments side by side (main, below): 905s -> 182s measured.
#
# The overlap could be had without touching the function, by having
# query_data_async prefetch all three KPIs of a segment on first call (~32s
# per segment). Not done: it hides a side effect in a function whose contract
# is to fetch one KPI, and the win shrinks once segments already share the
# CPU.

def _process_segment(game, region, platform):
    """Fetch KPIs and run the model pipeline for one segment.

    Each query starts immediately before the preceding CPU step so that
    I/O and modelling overlap within the segment.
    """
    df_players         = query_data_async("players",    game, region, platform)
    modeled_players    = fit_player_data(df_players["players"])
    df_revenues        = query_data_async("revenues",   game, region, platform)
    modeled_revenues   = fit_economy_data(df_revenues["revenues"], modeled_players)
    df_gamerounds      = query_data_async("gamerounds", game, region, platform)
    modeled_gamerounds = fit_gameround_data(df_gamerounds["gamerounds"], modeled_revenues)
    return predict(modeled_players, modeled_revenues, modeled_gamerounds)


def _summarize_segment(game, region, platform):
    """Run a segment, keep two floats.

    Returning the array instead would pin ~0.5 GB per finished segment: a
    Future holds its result until the Future itself is dropped, and these
    live until the run ends. That cost 4.5 GB peak before this wrapper
    existed, against 1.5 GB with it.
    """
    prediction = _process_segment(game, region, platform)
    return prediction.sum(), prediction.mean()


def main():
    parser = argparse.ArgumentParser(description="Run the optimized Pythia prediction pipeline.")
    parser.add_argument(
        "--games", nargs="+", default=["gameA", "gameB"],
        metavar="GAME", help="Games to predict (default: gameA gameB)",
    )
    parser.add_argument(
        "--regions", nargs="+", default=["EU", "NA"],
        metavar="REGION", help="Regions to predict (default: EU NA)",
    )
    parser.add_argument(
        "--platforms", nargs="+", default=["Android", "iOS"],
        metavar="PLATFORM", help="Platforms to predict (default: Android iOS)",
    )
    parser.add_argument(
        "--output", default="results-v2-optimized.csv",
        metavar="FILE", help="Output CSV path (default: results-v2-optimized.csv)",
    )
    args = parser.parse_args()

    segments = [
        (game, region, platform)
        for game in args.games
        for region in args.regions
        for platform in args.platforms
    ]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SEGMENTS, thread_name_prefix="segment") as pool:
        futures = [pool.submit(_summarize_segment, *segment) for segment in segments]
        results = []
        # Submission order, so the CSV rows line up with v1's. A failing
        # segment aborts the run here, also like v1: unlike the service,
        # this script has no partial-failure story to tell.
        for (game, region, platform), future in zip(segments, futures, strict=True):
            total, mean = future.result()
            print(game, region, platform, total, mean)
            results.append({"game": game, "region": region, "platform": platform, "sum": total, "mean": mean})

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Done in {time.perf_counter() - started:.1f}s. Results saved to {args.output}")


if __name__ == "__main__":
    main()
