"""
Player modelling library.

This module is owned by the Data Science team and must not be modified.
It exposes four functions that form a sequential modelling pipeline:

    fit_player_data → fit_economy_data → fit_gameround_data → predict

Each function is CPU-intensive.  See individual docstrings for runtimes.
"""
import os
import time

# Limit each process to 1 BLAS thread so num_cpus processes == num_cpus cores.
# Must be set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import multiprocessing

_WORK_UNIT_DIM   = 200
_ITERS_PER_SECOND = None


def _work_unit(a):
    return float(np.dot(a, a).sum())


def _calibrate(duration=1.0):
    """Measure work units per second (single thread).  Runs once per session."""
    global _ITERS_PER_SECOND
    a      = np.random.default_rng(0).random((_WORK_UNIT_DIM, _WORK_UNIT_DIM))
    count  = 0
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        _work_unit(a)
        count += 1
    _ITERS_PER_SECOND = max(count, 1)
    return _ITERS_PER_SECOND


def _worker(target_iters):
    a = np.random.default_rng(0).random((_WORK_UNIT_DIM, _WORK_UNIT_DIM))
    for _ in range(target_iters):
        _work_unit(a)


def _cpu_intensive(num_seconds, num_cpus, output_gb, seed=42):
    """Run CPU-intensive work across num_cpus cores for approximately num_seconds."""
    if _ITERS_PER_SECOND is None:
        _calibrate()
    target_iters = max(int(_ITERS_PER_SECOND * num_seconds), 1)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.map(_worker, [target_iters] * num_cpus)
    num_elements = max(int(output_gb * 1e9 / np.dtype(float).itemsize), 1)
    n_rows = max(int(np.sqrt(num_elements)), 1)
    n_cols = max(num_elements // n_rows, 1)
    return np.random.default_rng(seed).random((n_rows, n_cols))


def benchmark(seconds_list=(1, 2, 5), num_cpus=1):
    """Compare requested wall-clock seconds against actual execution time.

    Useful for verifying calibration on your machine.

    Args:
        seconds_list: Iterable of requested durations to test.
        num_cpus: Number of parallel workers.

    Returns:
        List of dicts with keys ``'requested'``, ``'actual'``, ``'ratio'``.
    """
    col = 14
    header    = f"{'requested (s)':>{col}}  {'actual (s)':>{col}}  {'ratio':>{col}}"
    separator = "-" * len(header)
    print(header)
    print(separator)
    results = []
    for sec in seconds_list:
        t0      = time.perf_counter()
        _cpu_intensive(num_seconds=sec, num_cpus=num_cpus, output_gb=0.001)
        elapsed = time.perf_counter() - t0
        ratio   = elapsed / sec
        results.append({"requested": sec, "actual": round(elapsed, 3), "ratio": round(ratio, 3)})
        print(f"{sec:>{col}}  {elapsed:>{col}.3f}  {ratio:>{col}.3f}")
    return results


def fit_player_data(data, num_cpus=4):
    """Fit a player behaviour model.

    Runtime ~5 s | output ~0.05 GB

    Args:
        data:     pandas.Series of player KPI values.
        num_cpus: CPU cores to use internally.

    Returns:
        numpy.ndarray — fitted model output.
    """
    print(f"[modeling] fit_player_data starting ({num_cpus} cores)")
    seed   = int(abs(data.sum()) * 1e4) % (2**31)
    result = _cpu_intensive(num_seconds=5, num_cpus=num_cpus, output_gb=0.05, seed=seed)
    print("[modeling] fit_player_data done")
    return result


def fit_economy_data(data, modeled_players, num_cpus=4):
    """Fit an in-game economy model.

    Depends on modeled_players — must be called after fit_player_data.
    Runtime ~5 s | output ~0.05 GB

    Args:
        data:             pandas.Series of revenue KPI values.
        modeled_players:  Output of fit_player_data.
        num_cpus:         CPU cores to use internally.

    Returns:
        numpy.ndarray — fitted model output.
    """
    print(f"[modeling] fit_economy_data starting ({num_cpus} cores)")
    seed   = int(abs(data.sum() + modeled_players.sum()) * 1e4) % (2**31)
    result = _cpu_intensive(num_seconds=5, num_cpus=num_cpus, output_gb=0.05, seed=seed)
    print("[modeling] fit_economy_data done")
    return result


def fit_gameround_data(data, modeled_revenues, num_cpus=4):
    """Fit a game-round engagement model.

    Depends on modeled_revenues — must be called after fit_economy_data.
    Runtime ~4 s | output ~0.1 GB

    Args:
        data:              pandas.Series of gameround KPI values.
        modeled_revenues:  Output of fit_economy_data.
        num_cpus:          CPU cores to use internally.

    Returns:
        numpy.ndarray — fitted model output.
    """
    print(f"[modeling] fit_gameround_data starting ({num_cpus} cores)")
    seed   = int(abs(data.sum() + modeled_revenues.sum()) * 1e4) % (2**31)
    result = _cpu_intensive(num_seconds=4, num_cpus=num_cpus, output_gb=0.1, seed=seed)
    print("[modeling] fit_gameround_data done")
    return result


def predict(modeled_players, modeled_revenues, modeled_gamerounds, num_cpus=4):
    """Combine fitted models into a forward prediction.

    Runtime ~3 s | output ~0.5 GB

    Args:
        modeled_players:    Output of fit_player_data.
        modeled_revenues:   Output of fit_economy_data.
        modeled_gamerounds: Output of fit_gameround_data.
        num_cpus:           CPU cores to use internally.

    Returns:
        numpy.ndarray — prediction values.
    """
    print(f"[modeling] predict starting ({num_cpus} cores)")
    seed   = int(abs(modeled_players.sum() + modeled_revenues.sum() + modeled_gamerounds.sum()) * 1e4) % (2**31)
    result = _cpu_intensive(num_seconds=3, num_cpus=num_cpus, output_gb=0.5, seed=seed)
    print("[modeling] predict done")
    return result
