"""
Pythia — player prediction pipeline.

Entry point for the library.  You may adapt the interface of this file
(see README.md), but do not modify pythia_library_v1/bigquery.py or
pythia_library_v1/modeling.py.

Usage
-----
python scripts/pythia-prediction-v1.py
python scripts/pythia-prediction-v1.py --games gameA gameB --regions EU --platforms Android
python scripts/pythia-prediction-v1.py --output results-v1.csv
"""
import argparse
import pandas as pd

from pythia_library_v1.modeling import fit_player_data, fit_economy_data, fit_gameround_data, predict
from pythia_library_v1.bigquery import query_data


def _get_data(games, regions, platforms):
    """Fetch and merge KPI data for every (game, region, platform) combination."""
    segments = []
    for game in games:
        for region in regions:
            for platform in platforms:
                df_players    = query_data("players",    game, region, platform)
                df_revenues   = query_data("revenues",   game, region, platform)
                df_gamerounds = query_data("gamerounds", game, region, platform)
                df = pd.merge(
                    df_players,
                    df_revenues[["dt", "game", "region", "platform", "revenues"]],
                    on=["dt", "game", "region", "platform"],
                )
                df = pd.merge(
                    df,
                    df_gamerounds[["dt", "game", "region", "platform", "gamerounds"]],
                    on=["dt", "game", "region", "platform"],
                )
                segments.append(df)
    return pd.concat(segments, ignore_index=True)


def pythia_oracle(segment_data):
    """Run the full modelling pipeline for a single segment.

    Args:
        segment_data: DataFrame for one (game, region, platform) combination,
                      with columns ``players``, ``revenues``, ``gamerounds``.

    Returns:
        numpy.ndarray — prediction values for the segment.
    """
    modeled_players    = fit_player_data(segment_data["players"])
    modeled_revenues   = fit_economy_data(segment_data["revenues"], modeled_players)
    modeled_gamerounds = fit_gameround_data(segment_data["gamerounds"], modeled_revenues)
    return predict(modeled_players, modeled_revenues, modeled_gamerounds)


def main():
    parser = argparse.ArgumentParser(description="Run the Pythia prediction pipeline.")
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
        "--output", default="results-v1.csv",
        metavar="FILE", help="Output CSV path (default: results-v1.csv)",
    )
    args = parser.parse_args()

    data    = _get_data(args.games, args.regions, args.platforms)
    results = []

    for game in args.games:
        for region in args.regions:
            for platform in args.platforms:
                segment_data = data[
                    (data["game"] == game) &
                    (data["region"] == region) &
                    (data["platform"] == platform)
                ]
                result = pythia_oracle(segment_data)
                print(game, region, platform, result.sum(), result.mean())
                results.append({
                    "game": game, "region": region, "platform": platform,
                    "sum": result.sum(), "mean": result.mean(),
                })

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
