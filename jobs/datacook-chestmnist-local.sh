#!/usr/bin/env bash
# ChestMNIST multi-label DataCook runner.
# Project: DataCook
# Created: 2026-07-22
# Commit:  1566c034
#
# Validation run (default, seed 0 / epsilon 8):
#   CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook RUN_MODE=smoke bash jobs/datacook-chestmnist-local.sh
# Three-seed main experiment plus post-hoc analysis:
#   CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook RUN_MODE=full bash jobs/datacook-chestmnist-local.sh

set -euo pipefail

JOB_NAME="datacook-chestmnist"
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
  run_logged python run_datacook_pipeline.py \
    --config configs/pipelines/chestmnist.yaml
else
  # The completed smoke run is automatically reused.
  run_logged python run_datacook_pipeline.py \
    --config configs/pipelines/chestmnist.yaml \
    --seeds 0 1 2 \
    --epsilons 8

  run_logged python prepare_revision_posthoc.py \
    --suite-config configs/pipelines/medmnist_multilabel.yaml \
    --seeds 0 1 2 \
    --epsilons 8 \
    --artifact-root artifacts/revision_pipeline_multilabel \
    --config-dir configs/revision/chestmnist/generated \
    --output-suffix chestmnist \
    --attacks-epsilon 8 \
    --runtime-seed 0 \
    --runtime-epsilon 8 \
    --strict

  run_logged python run_revision_experiments.py \
    --config configs/revision/chestmnist/generated.yaml \
    --tasks pseudo image_quality runtime attacks finetune_attacks finetune_budget_attacks \
    --device cuda
fi

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)"                            | tee -a "$LOG_FILE"
echo "Main: results/revision_pipeline_multilabel/" | tee -a "$LOG_FILE"
echo "Post-hoc suffix: *_chestmnist.csv"       | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
