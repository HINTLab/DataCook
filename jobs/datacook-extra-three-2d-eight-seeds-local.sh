#!/usr/bin/env bash
# Second-server job: three additional 2D datasets, eight seeds, five epsilons.
# Project: DataCook
# Created: 2026-07-21
# Commit:  1566c034
#
# Example: CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook bash jobs/datacook-extra-three-2d-eight-seeds-local.sh

set -euo pipefail

JOB_NAME="datacook-extra-three-2d-eight-seeds"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

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

# Smoke test. These three runs are reused by the full grid below.
run_logged python run_datacook_suite.py \
  --config configs/pipelines/medmnist_extra_three.yaml \
  --seeds 0 \
  --epsilons 8

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Smoke test completed; SMOKE_ONLY=1, stopping before the full grid." | tee -a "$LOG_FILE"
  exit 0
fi

# Full grid: 3 datasets x 8 seeds x 5 epsilon values = 120 runs.
run_logged python run_datacook_suite.py \
  --config configs/pipelines/medmnist_extra_three.yaml \
  --seeds 0 1 2 3 4 5 6 7 \
  --epsilons 2 4 8 12 16

# Export all post-hoc artifacts. Adaptive attacks are generated only for epsilon=8.
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist_extra_three.yaml \
  --artifact-root artifacts/revision_pipeline_extra2d \
  --config-dir configs/revision/extensions/extra2d/generated \
  --attacks-epsilon 8 \
  --runtime-seed 0 \
  --runtime-epsilon 8 \
  --finetune-epochs 10 \
  --budget-sizes 25 50 \
  --output-suffix extra2d \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/extensions/extra2d/generated.yaml \
  --tasks pseudo image_quality runtime attacks finetune_attacks finetune_budget_attacks \
  --device cuda

run_logged python -m analysis.finalize_revision_results \
  --config configs/revision/final_analysis_extra2d_8seeds.yaml

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"
echo "Main results: results/revision_pipeline_extra2d/" | tee -a "$LOG_FILE"
echo "Post-hoc shards: results/revision_pipeline/posthoc/*_extra2d.csv" | tee -a "$LOG_FILE"
echo "Final analysis: results/revision_pipeline_extra2d/final_8seeds/" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
