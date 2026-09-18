#!/usr/bin/env bash
# Run the optimized pipeline (pythia_library_v2) and save results-v2-optimized.csv.
# Pass additional arguments to override games, regions, platforms, or output path:
#   ./run-v2-optimized.sh --games gameA --regions EU --platforms Android
set -euo pipefail
PYTHONPATH="$(pwd)" python scripts/pythia-prediction-v2-optimized.py "$@"
