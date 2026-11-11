#!/usr/bin/env bash
# 3D pseudo-label and runtime post-hoc analysis for completed DataCook runs.
# Project: DataCook
# Created: 2026-07-23
# Commit:  1566c034
#
# Foreground run inside an existing screen session:
#   CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook bash jobs/datacook-3d-pseudo-runtime-local.sh

set -euo pipefail

JOB_NAME="datacook-3d-pseudo-runtime"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GIT_COMMIT="1566c034"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"

echo "=========================================" | tee "$LOG_FILE"
echo "Job:    $JOB_NAME"                       | tee -a "$LOG_FILE"
echo "Start:  $(date)"                         | tee -a "$LOG_FILE"
echo "Commit: $GIT_COMMIT"                     | tee -a "$LOG_FILE"
echo "GPU:    $CUDA_VISIBLE_DEVICES"           | tee -a "$LOG_FILE"
echo "Log:    $LOG_FILE"                       | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "conda command not found; using the currently active Python environment." | tee -a "$LOG_FILE"
fi

cd "$PROJECT_ROOT"

run_logged() {
  echo "Running: $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

# Reuse all 36 completed 3D runs. This only exports test NPZ artifacts and
# generates pseudo/runtime configs; it does not retrain any model.
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist3d_four.yaml \
  --artifact-root artifacts/revision_pipeline_3d \
  --result-root results/revision_pipeline_3d \
  --config-dir configs/revision/3d/generated \
  --tasks pseudo runtime \
  --runtime-seed 0 \
  --runtime-epsilon 8 \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/3d/generated.yaml \
  --tasks pseudo runtime \
  --device cuda

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)"                            | tee -a "$LOG_FILE"
echo "Pseudo:  results/revision_pipeline_3d/posthoc/pseudo_label_quality.csv" | tee -a "$LOG_FILE"
echo "Runtime: results/revision_pipeline_3d/posthoc/runtime_summary.csv"      | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
