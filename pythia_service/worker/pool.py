"""
Shared segment-execution pool.

Revised decision (see conversation): originally planned as a
ProcessPoolExecutor, on the assumption that the fit steps were CPU-bound
inside *our* process and needed real processes to bypass the GIL.

That assumption was wrong for this library. The README states the modeling
functions "each spawn an internal process pool" — the CPU-heavy work already
happens in subprocesses managed by pythia_library itself. From our process's
point of view, calling fit_player_data(...) is a *blocking call that waits on
its own subprocesses*, which releases the GIL exactly like blocking I/O does.

Two consequences:
1. A ProcessPoolExecutor here would be actively wrong, not just suboptimal —
   worker processes created by multiprocessing are daemonic by default, and
   daemonic processes cannot spawn children. Calling fit_player_data from
   inside one would raise AssertionError at runtime.
2. A ThreadPoolExecutor is sufficient for our layer: segment concurrency is
   about overlapping blocking waits (I/O fetch + waiting on the library's own
   subprocess pool), not about doing CPU work ourselves.

The real resource constraint doesn't disappear, it just moves: each segment
running through the library uses up to 4 cores internally. If we let S
segments run concurrently, actual CPU usage is S * 4 cores. So pool size is
still the admission-control lever (same intent as the original decision,
different mechanism) — sized so that S * 4 doesn't exceed the machine's cores.
"""

import os
from concurrent.futures import ThreadPoolExecutor

CORES_PER_SEGMENT = 4  # fixed cost stated in the README for the fit steps


def compute_pool_size(available_cores: int | None = None) -> int:
    """Max number of segments allowed to run concurrently.

    Floor of 1: even on a machine with fewer than 4 cores, we still want to
    process segments one at a time rather than refuse to start.
    """
    cores = available_cores if available_cores is not None else (os.cpu_count() or CORES_PER_SEGMENT)
    return max(1, cores // CORES_PER_SEGMENT)


class SegmentPool:
    """Thin wrapper around ThreadPoolExecutor with explicit lifecycle
    (create once at app startup, shut down once at app shutdown) instead of
    a module-level global — makes ownership explicit and testable, and
    avoids a pool silently surviving across app reloads in dev.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._max_workers = max_workers if max_workers is not None else compute_pool_size()
        self._executor: ThreadPoolExecutor | None = None

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def start(self) -> None:
        if self._executor is not None:
            return  # idempotent: calling start() twice is a no-op, not an error
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="pythia-segment"
        )

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=wait)
        self._executor = None

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError("SegmentPool.start() must be called before submitting work")
        return self._executor
