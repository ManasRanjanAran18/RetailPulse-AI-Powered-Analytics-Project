#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rerun_all.sh - rebuild EVERYTHING from scratch:
#   data -> ETL -> models -> forecasts -> reports, then re-execute all notebooks.
#
# Usage:
#   ./rerun_all.sh              # full run (needs requirements-full.txt)
#   ./rerun_all.sh --light      # skip Prophet/LSTM and notebooks (fast)
#   ./rerun_all.sh --no-notebooks
#   ./rerun_all.sh --regenerate # also regenerate the synthetic raw data
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

LIGHT=0; NOTEBOOKS=1; REGEN=""
for arg in "$@"; do
  case "$arg" in
    --light) LIGHT=1; NOTEBOOKS=0 ;;
    --no-notebooks) NOTEBOOKS=0 ;;
    --regenerate) REGEN="--regenerate" ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD"
export TF_CPP_MIN_LOG_LEVEL=3

echo "==> 1/3 Unit tests";            "$PY" -m pytest tests -q
echo "==> 2/3 Pipeline (ETL, segmentation, churn, forecasting, inventory, drift)"
if [ "$LIGHT" = "1" ]; then
  "$PY" scripts/run_pipeline.py --light $REGEN
else
  "$PY" scripts/run_pipeline.py --mlflow $REGEN
fi

if [ "$NOTEBOOKS" = "1" ]; then
  echo "==> 3/3 Executing notebooks in order"
  for nb in notebooks/*.ipynb; do
    echo "    - $nb"
    "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=600 "$nb" >/dev/null
  done
else
  echo "==> 3/3 Notebooks skipped"
fi
echo "Done. Launch the dashboard with:  streamlit run dashboard/app.py"
