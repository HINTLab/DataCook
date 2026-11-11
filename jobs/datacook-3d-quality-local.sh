#!/usr/bin/env bash
# 3D image-quality analysis for all completed MedMNIST3D DataCook runs.
# Project: DataCook
# Created: 2026-07-24
# Commit:  1566c03404
#
# This is CPU post-processing and does not retrain any model.
# Foreground run inside an existing screen session:
#   CONDA_ENV=datacook bash jobs/datacook-3d-quality-local.sh

set -euo pipefail

JOB_NAME="datacook-3d-quality"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/results/revision_pipeline_3d/posthoc"
CONDA_ENV="${CONDA_ENV:-datacook}"
GIT_COMMIT="1566c03404"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "conda command not found; using the currently active Python environment."
fi

cd "$PROJECT_ROOT"

echo "=========================================" | tee "$LOG_FILE"
echo "Job:    $JOB_NAME"                       | tee -a "$LOG_FILE"
echo "Start:  $(date)"                         | tee -a "$LOG_FILE"
echo "Commit: $GIT_COMMIT"                     | tee -a "$LOG_FILE"
echo "Device: CPU"                             | tee -a "$LOG_FILE"
echo "Log:    $LOG_FILE"                       | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

run_logged() {
  echo "Running: $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

# Re-export cooked volumes from the current perturbations. --force is required
# so a later Synapse rerun cannot leave a stale cooked_test.npz in the analysis.
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist3d_four.yaml \
  --artifact-root artifacts/revision_pipeline_3d \
  --result-root results/revision_pipeline_3d \
  --config-dir configs/revision/3d/generated \
  --tasks image_quality \
  --force \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/3d/generated.yaml \
  --tasks image_quality

QUALITY_FILE="$OUTPUT_DIR/quality_metrics.csv"
EXPECTED_LINES=37
ACTUAL_LINES="$(wc -l < "$QUALITY_FILE" | tr -d ' ')"
if [[ "$ACTUAL_LINES" -ne "$EXPECTED_LINES" ]]; then
  echo "Expected 36 quality rows plus header, found $ACTUAL_LINES lines." | tee -a "$LOG_FILE" >&2
  exit 3
fi

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)"                            | tee -a "$LOG_FILE"
echo "Rows: 36 (4 datasets x 3 seeds x 3 epsilon values)" | tee -a "$LOG_FILE"
echo "Metrics: 3D SSIM, PSNR, L2 norm, Linf norm"          | tee -a "$LOG_FILE"
echo "LPIPS/FID: not computed for 3D volumes"               | tee -a "$LOG_FILE"
echo "Output: results/revision_pipeline_3d/posthoc/quality_metrics.csv" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
