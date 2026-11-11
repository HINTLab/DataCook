#!/usr/bin/env bash
# Local experiment runner generated from the run-experiment workflow.
# Project: DataCook
# Created: 2026-07-20
# Commit:  1566c034
#
# Run: CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook bash jobs/datacook-medmnist-additional-two-local.sh

set -euo pipefail

JOB_NAME="datacook-medmnist-additional-two"
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

RUN_COMMAND=(python run_datacook_suite.py --config configs/pipelines/medmnist_additional_two.yaml)
echo "Running: ${RUN_COMMAND[*]}" | tee -a "$LOG_FILE"
echo "-----------------------------------------" | tee -a "$LOG_FILE"

set +e
"${RUN_COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End:  $(date)" | tee -a "$LOG_FILE"
echo "Exit: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

exit "$EXIT_CODE"
