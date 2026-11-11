#!/usr/bin/env bash
# CPU-only merge after both 2D experiment servers have been synchronized.

set -euo pipefail

JOB_NAME="datacook-merge-six-2d-results"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

cd "$PROJECT_ROOT"

run_logged() {
  echo "Running: $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

run_logged python -m analysis.merge_result_shards \
  --config configs/revision/merge_2d_six_datasets.yaml

run_logged python -m analysis.finalize_revision_results \
  --config configs/revision/final_analysis_2d_six_datasets.yaml

echo "Final six-dataset results: results/revision_pipeline_6datasets/" | tee -a "$LOG_FILE"
