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
# TODO: add any imports you need

from pythia_library_v2.modeling import fit_player_data, fit_economy_data, fit_gameround_data, predict
from pythia_library_v2.bigquery import query_data


# TODO: implement query_data_async
# It should start the query in a background thread immediately (non-blocking)
# and return a handle whose ["column"] access blocks until the data is ready.
def query_data_async(kpi_name, game, region, platform):
    raise NotImplementedError


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


# TODO: implement main()
# - Same argument interface as scripts/pythia-prediction.py
# - Process segments concurrently
# - Default output: results-v2-optimized.csv
def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
