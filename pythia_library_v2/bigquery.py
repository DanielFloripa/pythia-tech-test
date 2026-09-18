"""
BigQuery client — v2 (updated latencies).

The Data Engineering team has improved data quality at the cost of higher
query latency.  This file replaces bigquery.py in the candidate's library.
"""
import random
import time
import pandas as pd

_DATASET_CONFIG = {
    "players":    {"seconds": 15},
    "revenues":   {"seconds": 10},
    "gamerounds": {"seconds": 8},
}

_ROW_COUNT = {
    ("gameA", "EU",  "iOS"):     1200,
    ("gameA", "EU",  "Android"): 980,
    ("gameA", "NA",  "iOS"):     1500,
    ("gameA", "NA",  "Android"): 1350,
    ("gameB", "EU",  "iOS"):     870,
    ("gameB", "EU",  "Android"): 760,
    ("gameB", "NA",  "iOS"):     1100,
    ("gameB", "NA",  "Android"): 950,
}


def query_data(kpi_name, game, region, platform):
    """Fetch a KPI table from BigQuery for the given segment.

    Args:
        kpi_name:  One of ``'players'``, ``'revenues'``, ``'gamerounds'``.
        game:      Game identifier string.
        region:    Region identifier string.
        platform:  Platform identifier string.

    Returns:
        pandas.DataFrame with columns ``[kpi_name, 'dt', 'game', 'region', 'platform']``.

    KPI names and their approximate query latencies:
        - ``'players'``    : 15 s
        - ``'revenues'``   : 10 s
        - ``'gamerounds'`` :  8 s

    Raises:
        KeyError: if the (game, region, platform) combination is not configured.
    """
    config = _DATASET_CONFIG[kpi_name]
    print(f"[bigquery] querying {kpi_name} for {game}/{region}/{platform} ...")
    time.sleep(config["seconds"])
    print(f"[bigquery] {kpi_name} received")

    rows = _ROW_COUNT.get((game, region, platform))
    if rows is None:
        raise KeyError(
            f"No data configured for: game={game!r}, region={region!r}, platform={platform!r}"
        )

    rng   = random.Random(42)
    dates = pd.date_range(end="2024-12-31", periods=rows, freq="D")
    return pd.DataFrame({
        kpi_name:   [rng.random() for _ in range(rows)],
        "dt":       dates,
        "game":     [game]     * rows,
        "region":   [region]   * rows,
        "platform": [platform] * rows,
    })
