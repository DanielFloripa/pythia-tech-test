#!/usr/bin/env bash
# Run the sequential pipeline (pythia_library_v2) and save results-v2.csv.
# Use this as the timing baseline to compare against ./run-v2-optimized.sh.
set -euo pipefail
PYTHONPATH="$(pwd)" python scripts/pythia-prediction-v2.py "$@"
