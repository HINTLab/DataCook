#!/usr/bin/env bash
# Representative 224x224 MedMNIST+ experiment runner.
# Project: DataCook
# Created: 2026-07-22
# Commit:  1566c034
#
# Smoke test (default):
#   CUDA_VISIBLE_DEVICES=0 RUN_MODE=smoke bash jobs/datacook-224-three-local.sh
# Full 3-dataset, 3-seed run:
#   CUDA_VISIBLE_DEVICES=0 RUN_MODE=full bash jobs/datacook-224-three-local.sh

set -euo pipefail

JOB_NAME="datacook-224-three"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"
RUN_MODE="${RUN_MODE:-smoke}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GIT_COMMIT="1566c034"

if [[ "$RUN_MODE" != "smoke" && "$RUN_MODE" != "full" ]]; then
  echo "RUN_MODE must be smoke or full, got: $RUN_MODE" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-${RUN_MODE}-$(date +%Y%m%d-%H%M%S).log"

echo "=========================================" | tee "$LOG_FILE"
echo "Job:    $JOB_NAME"                       | tee -a "$LOG_FILE"
echo "Mode:   $RUN_MODE"                       | tee -a "$LOG_FILE"
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

if [[ "$RUN_MODE" == "smoke" ]]; then
  # Validate downloading, 224 loading, cooking, protected training and metrics
  # on the smallest dataset before committing the GPU to all nine runs.
  run_logged python run_datacook_pipeline.py \
    --config configs/pipelines/breastmnist224.yaml \
    --seeds 0 \
    --epsilons 8
else
  # The smoke result is automatically detected and reused.
  run_logged python run_datacook_suite.py \
    --config configs/pipelines/medmnist_224_three.yaml

  # Export paired 224 artifacts and generate post-hoc configs without touching
  # the existing 28x28 artifacts/results.
  run_logged python prepare_revision_posthoc.py \
    --suite-config configs/pipelines/medmnist_224_three.yaml \
    --artifact-root artifacts/revision_pipeline_224 \
    --config-dir configs/revision/224/generated \
    --output-suffix 224 \
    --attacks-epsilon 8 \
    --runtime-seed 0 \
    --runtime-epsilon 8 \
    --strict

  run_logged python run_revision_experiments.py \
    --config configs/revision/224/generated.yaml \
    --tasks pseudo image_quality runtime attacks \
    --device cuda
fi

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)"                            | tee -a "$LOG_FILE"
echo "Main: results/revision_pipeline_224/"    | tee -a "$LOG_FILE"
echo "Post-hoc: results/revision_pipeline/posthoc/*_224.csv" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
