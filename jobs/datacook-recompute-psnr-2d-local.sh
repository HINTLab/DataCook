#!/usr/bin/env bash
# Recompute 2D image quality after the aggregate-MSE PSNR fix, then rebuild tables.
# Project: DataCook
# Created: 2026-07-22
# Commit:  1566c034
#
# Run in the current screen session:
#   CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook bash jobs/datacook-recompute-psnr-2d-local.sh

set -euo pipefail

JOB_NAME="datacook-recompute-psnr-2d"
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

# Recompute all 120 quality rows. No training/checkpoint is touched.
run_logged python run_revision_experiments.py \
  --config configs/revision/generated.yaml \
  --tasks image_quality \
  --device cuda

run_logged python run_revision_experiments.py \
  --config configs/revision/extensions/seed3-7-all/generated.yaml \
  --tasks image_quality \
  --device cuda

# Rebuild the extended quality shard and all final paper-ready tables.
run_logged python -m analysis.merge_result_shards \
  --config configs/revision/merge_2d_8seeds.yaml

run_logged python -m analysis.finalize_revision_results \
  --config configs/revision/final_analysis_2d_8seeds.yaml

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)"                            | tee -a "$LOG_FILE"
echo "Quality: results/revision_pipeline/posthoc/quality_metrics_extended_2d_8seeds.csv" | tee -a "$LOG_FILE"
echo "Final:   results/revision_pipeline/final_2d_8seeds/" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
