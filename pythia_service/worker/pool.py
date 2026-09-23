"""
Where segments actually run.

Threads, not processes, and that is not the obvious choice. pythia_library's
fit functions open their own multiprocessing.Pool, so the CPU work already
happens in child processes; from here a fit is a blocking wait, which frees
the GIL exactly like I/O does. Putting this in a ProcessPoolExecutor would
also break outright: multiprocessing workers are daemonic and daemonic
processes cannot have children.

The CPU limit does not disappear, it moves. Each segment burns up to 4 cores
inside the library, so the number of concurrent segments is the only knob we
have over machine load.
"""

import os
from concurrent.futures import ThreadPoolExecutor

CORES_PER_SEGMENT = 4  # num_cpus the library passes to each fit


def compute_pool_size(available_cores: int | None = None) -> int:
    """How many segments may run at once. Floor of 1, so a 2-core box still
    works, one segment at a time."""
    cores = available_cores if available_cores is not None else (os.cpu_count() or CORES_PER_SEGMENT)
    return max(1, cores // CORES_PER_SEGMENT)


class SegmentPool:
    """ThreadPoolExecutor with an explicit lifecycle. A module-level global
    would start a pool on import and survive reloads in dev; here the app
    owns it and tests can build their own.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._max_workers = max_workers if max_workers is not None else compute_pool_size()
        self._executor: ThreadPoolExecutor | None = None

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def start(self) -> None:
        if self._executor is not None:
            return  # start() twice is a no-op, not an error
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="pythia-segment"
        )

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._executor = None

    @property
    def is_running(self) -> bool:
        return self._executor is not None

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError("SegmentPool.start() must be called before submitting work")
        return self._executor
