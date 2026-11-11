#!/usr/bin/env bash
# Complete the current three 2D datasets with eight seeds and targeted attacks.
# Project: DataCook
# Created: 2026-07-21
# Commit:  1566c034
#
# Example: CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook bash jobs/datacook-2d-eight-seeds-local.sh

set -euo pipefail

JOB_NAME="datacook-2d-eight-seeds"
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

for required in \
  results/revision_pipeline/posthoc/adaptive_attack_results_all.csv \
  results/revision_pipeline/posthoc/adaptive_finetune_budget_results.csv \
  results/revision_pipeline/posthoc/quality_metrics.csv \
  results/revision_pipeline/posthoc/pseudo_label_quality.csv; do
  if [[ ! -f "$required" ]]; then
    echo "Missing prerequisite: $required" | tee -a "$LOG_FILE"
    exit 2
  fi
done

# Phase 1: 75 new main runs. Existing seed 0-2 runs are untouched and reused.
run_logged python run_datacook_suite.py \
  --config configs/pipelines/medmnist_three.yaml \
  --seeds 3 4 5 6 7 \
  --epsilons 2 4 8 12 16

# Phase 2: quality and pseudo-label outputs for every new seed/epsilon run.
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_three.yaml \
  --seeds 3 4 5 6 7 \
  --config-dir configs/revision/extensions/seed3-7-all/generated \
  --output-suffix seed3-7-all \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/extensions/seed3-7-all/generated.yaml \
  --tasks pseudo image_quality \
  --device cuda

# Phase 3: epsilon=8 attacks for the five newly added seeds.
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_three.yaml \
  --seeds 3 4 5 6 7 \
  --epsilons 8 \
  --attacks-epsilon 8 \
  --finetune-epochs 10 \
  --budget-sizes 25 50 \
  --config-dir configs/revision/extensions/seed3-7-eps8/generated \
  --output-suffix seed3-7-eps8 \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/extensions/seed3-7-eps8/generated.yaml \
  --tasks attacks finetune_attacks finetune_budget_attacks \
  --device cuda

# Phase 4: RetinaMNIST epsilon=12 attacks for all eight seeds.
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_three.yaml \
  --datasets RetinaMNIST \
  --seeds 0 1 2 3 4 5 6 7 \
  --epsilons 12 \
  --attacks-epsilon 12 \
  --finetune-epochs 10 \
  --budget-sizes 25 50 \
  --config-dir configs/revision/extensions/retina-eps12/generated \
  --output-suffix retina-eps12 \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/extensions/retina-eps12/generated.yaml \
  --tasks attacks finetune_attacks finetune_budget_attacks \
  --device cuda

# Phase 5: conflict-checked merge and final primary statistical analysis.
run_logged python -m analysis.merge_result_shards \
  --config configs/revision/merge_2d_8seeds.yaml

run_logged python -m analysis.finalize_revision_results \
  --config configs/revision/final_analysis_2d_8seeds.yaml

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"
echo "Main summary: results/revision_pipeline/summary/" | tee -a "$LOG_FILE"
echo "Extended post-hoc: results/revision_pipeline/posthoc/*extended_2d_8seeds.csv" | tee -a "$LOG_FILE"
echo "Final analysis: results/revision_pipeline/final_2d_8seeds/" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
