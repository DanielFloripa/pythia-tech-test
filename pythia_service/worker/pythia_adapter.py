"""
The only module that imports pythia_library. Switching the service to v2 is
an edit to the two imports below and nothing else.

No error handling or retries here on purpose: one function per pipeline
stage, library exceptions left to propagate. Deciding what a failure means
belongs to pipeline.py.
"""

import pandas as pd
from numpy.typing import NDArray

from pythia_library_v1 import bigquery, modeling

DEFAULT_NUM_CPUS = 4  # the library's own default


def fetch_players(game: str, region: str, platform: str) -> pd.Series:
    """~5s. Only the KPI column: the fits take a bare Series, and query_data
    returns it alongside dt/game/region/platform.
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
    """~5s. CPU-bound in the library's child processes, a blocking wait here."""
    return modeling.fit_player_data(players_data, num_cpus=num_cpus)


def fit_economy(
    revenues_data: pd.Series, modeled_players: NDArray, num_cpus: int = DEFAULT_NUM_CPUS
) -> NDArray:
    """~5s. Depends on fit_players' output."""
    return modeling.fit_economy_data(revenues_data, modeled_players, num_cpus=num_cpus)


def fit_gamerounds(
    gamerounds_data: pd.Series, modeled_economy: NDArray, num_cpus: int = DEFAULT_NUM_CPUS
) -> NDArray:
    """~4s. Takes fit_economy's output.

    The library calls this parameter `modeled_revenues`, which reads like raw
    revenue data. It is not, so the name here says what it holds.
    """
    return modeling.fit_gameround_data(gamerounds_data, modeled_economy, num_cpus=num_cpus)


def run_prediction(
    modeled_players: NDArray,
    modeled_economy: NDArray,
    modeled_gamerounds: NDArray,
    num_cpus: int = DEFAULT_NUM_CPUS,
) -> NDArray:
    """~3s. Called run_prediction so call sites read as an action rather
    than a re-exported `predict`.
    """
    return modeling.predict(modeled_players, modeled_economy, modeled_gamerounds, num_cpus=num_cpus)
