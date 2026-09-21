"""
Single point of contact with pythia_library_v1.

Nothing outside this file should `import pythia_library_v1` directly. If a
production deployment ever needs to switch to pythia_library_v2, this is the
one file that changes — everything else (worker/pipeline.py, the API layer)
depends only on the functions defined here.

Kept intentionally thin: no error handling, no retries, no store updates.
Each function does exactly one pipeline stage and lets the library's own
exceptions (e.g. KeyError from query_data on an unconfigured segment)
propagate as-is. worker/pipeline.py is responsible for catching per-stage
and recording which PipelineStage failed — that's an orchestration concern,
not an adapter concern.
"""

import pandas as pd
from numpy.typing import NDArray

from pythia_library_v1 import bigquery, modeling

DEFAULT_NUM_CPUS = 4  # matches the library's own default and the README's cost table


def fetch_players(game: str, region: str, platform: str) -> pd.Series:
    """~5s. Returns the 'players' column only — query_data's DataFrame also
    carries dt/game/region/platform columns that fit_player_data doesn't
    take (it wants a bare Series, per its docstring).
    """
    df = bigquery.query_data("players", game, region, platform)
    return df["players"]


def fetch_revenues(game: str, region: str, platform: str) -> pd.Series:
    """~3s."""
    df = bigquery.query_data("revenues", game, region, platform)
    return df["revenues"]


def fetch_gamerounds(game: str, region: str, platform: str) -> pd.Series:
    """~2s."""
    df = bigquery.query_data("gamerounds", game, region, platform)
    return df["gamerounds"]


def fit_players(players_data: pd.Series, num_cpus: int = DEFAULT_NUM_CPUS) -> NDArray:
    """~5s, CPU-bound inside the library (blocking from our side)."""
    return modeling.fit_player_data(players_data, num_cpus=num_cpus)


def fit_economy(
    revenues_data: pd.Series, modeled_players: NDArray, num_cpus: int = DEFAULT_NUM_CPUS
) -> NDArray:
    """~5s. Depends on fit_players' output."""
    return modeling.fit_economy_data(revenues_data, modeled_players, num_cpus=num_cpus)


def fit_gamerounds(
    gamerounds_data: pd.Series, modeled_economy: NDArray, num_cpus: int = DEFAULT_NUM_CPUS
) -> NDArray:
    """~4s. Depends on fit_economy's output.

    Note: the library's own parameter name for this is `modeled_revenues`,
    which is misleading — it's actually the output of fit_economy_data, not
    raw revenue data. Named `modeled_economy` here to match what it is, not
    what the library calls it (library itself is not ours to rename).
    """
    return modeling.fit_gameround_data(gamerounds_data, modeled_economy, num_cpus=num_cpus)


def run_prediction(
    modeled_players: NDArray,
    modeled_economy: NDArray,
    modeled_gamerounds: NDArray,
    num_cpus: int = DEFAULT_NUM_CPUS,
) -> NDArray:
    """~3s. Named run_prediction, not predict, to avoid shadowing the
    stdlib-adjacent builtin-sounding name and to keep call sites readable
    (`pythia_adapter.run_prediction(...)` reads clearer than a bare
    `predict(...)` re-export would).
    """
    return modeling.predict(modeled_players, modeled_economy, modeled_gamerounds, num_cpus=num_cpus)
