#!/usr/bin/env bash
# Run the Pythia pipeline (pythia_library_v1) and save results-v1.csv.
# Pass additional arguments to override games, regions, platforms, or output path:
#   ./run-v1.sh --games gameA --regions EU --platforms Android
set -euo pipefail
PYTHONPATH="$(pwd)" python scripts/pythia-prediction-v1.py "$@"
