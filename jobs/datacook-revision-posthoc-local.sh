#!/usr/bin/env bash
# Incremental post-hoc runner for completed DataCook main experiments.
# Project: DataCook
# Created: 2026-07-21
# Commit:  1566c034
#
# Run: CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook bash jobs/datacook-revision-posthoc-local.sh

set -euo pipefail

JOB_NAME="datacook-revision-posthoc"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"

echo "=========================================" | tee "$LOG_FILE"
echo "Job:    $JOB_NAME"                       | tee -a "$LOG_FILE"
echo "Start:  $(date)"                         | tee -a "$LOG_FILE"
echo "Commit: 1566c034"                        | tee -a "$LOG_FILE"
echo "Log:    $LOG_FILE"                       | tee -a "$LOG_FILE"
echo "GPU:    ${CUDA_VISIBLE_DEVICES:-all visible GPUs}" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "conda command not found; using the currently active Python environment." | tee -a "$LOG_FILE"
fi

cd "$PROJECT_ROOT"
export GIT_COMMIT="1566c034"

run_logged() {
  echo "Running: $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

set +e
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_three.yaml
EXIT_CODE=${PIPESTATUS[0]}

if [[ "$EXIT_CODE" -eq 0 ]]; then
  run_logged python run_revision_experiments.py \
    --config configs/revision/generated.yaml \
    --tasks pseudo image_quality runtime \
    --device cuda
  EXIT_CODE=${PIPESTATUS[0]}
fi

if [[ "$EXIT_CODE" -eq 0 ]]; then
  run_logged python run_revision_experiments.py \
    --config configs/revision/generated.yaml \
    --tasks attacks \
    --device cuda
  EXIT_CODE=${PIPESTATUS[0]}
fi
set -e

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End:  $(date)" | tee -a "$LOG_FILE"
echo "Exit: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

exit "$EXIT_CODE"
