#!/usr/bin/env bash
# Adaptive fine-tuning attacks for completed three-dataset DataCook runs.
# Project: DataCook
# Created: 2026-07-21
# Commit:  1566c034
#
# Example: CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook bash jobs/datacook-finetune-attacks-local.sh

set -euo pipefail

JOB_NAME="datacook-finetune-attacks"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"

echo "=========================================" | tee "$LOG_FILE"
echo "Job:    $JOB_NAME"                       | tee -a "$LOG_FILE"
echo "Start:  $(date)"                         | tee -a "$LOG_FILE"
echo "Commit: 1566c034"                        | tee -a "$LOG_FILE"
echo "Log:    $LOG_FILE"                       | tee -a "$LOG_FILE"
echo "GPU:    $CUDA_VISIBLE_DEVICES"           | tee -a "$LOG_FILE"
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

OLD_RESULTS="results/revision_pipeline/posthoc/adaptive_attack_results.csv"
if [[ ! -f "$OLD_RESULTS" ]]; then
  echo "Missing $OLD_RESULTS; run the inference attacks before fine-tuning attacks." | tee -a "$LOG_FILE"
  exit 2
fi

run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_three.yaml \
  --attacks-epsilon 8 \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/generated.yaml \
  --tasks finetune_attacks \
  --device cuda

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End:  $(date)" | tee -a "$LOG_FILE"
echo "Outputs:" | tee -a "$LOG_FILE"
echo "  results/revision_pipeline/posthoc/adaptive_finetune_results.csv" | tee -a "$LOG_FILE"
echo "  results/revision_pipeline/posthoc/adaptive_attack_results_all.csv" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
