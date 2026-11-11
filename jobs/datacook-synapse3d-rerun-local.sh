#!/usr/bin/env bash
# Foreground single-GPU rerun for all SynapseMNIST3D seeds and epsilon values.
# Project: DataCook
# Created: 2026-07-23
# Commit:  1566c03404
#
# This intentionally overwrites the existing SynapseMNIST3D primary outputs.
# Run inside an existing screen session:
#   CONDA_ENV=datacook bash jobs/datacook-synapse3d-rerun-local.sh 0

set -euo pipefail

JOB_NAME="datacook-synapse3d-rerun"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/results/revision_pipeline_3d"
CONDA_ENV="${CONDA_ENV:-datacook}"
GPU_ID="${1:-0}"
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
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=========================================" | tee "$LOG_FILE"
echo "Job:    $JOB_NAME"                       | tee -a "$LOG_FILE"
echo "Start:  $(date)"                         | tee -a "$LOG_FILE"
echo "Commit: $GIT_COMMIT"                     | tee -a "$LOG_FILE"
echo "GPU:    $GPU_ID"                         | tee -a "$LOG_FILE"
echo "Log:    $LOG_FILE"                       | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

run_logged() {
  echo "Running: $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

# --force reruns perturbation, protected-model training, and metrics for:
# seeds 0/1/2 x epsilon 8/16/32. Other 3D datasets are not retrained.
run_logged python -u run_datacook_pipeline.py \
  --config configs/pipelines/synapsemnist3d.yaml \
  --force

# Re-export derived artifacts and rebuild the complete four-dataset 3D
# pseudo-label/runtime tables so that no row still references old Synapse runs.
run_logged python prepare_revision_posthoc.py \
  --suite-config configs/pipelines/medmnist3d_four.yaml \
  --artifact-root artifacts/revision_pipeline_3d \
  --result-root results/revision_pipeline_3d \
  --config-dir configs/revision/3d/generated \
  --tasks pseudo runtime \
  --runtime-seed 0 \
  --runtime-epsilon 8 \
  --force \
  --strict

run_logged python run_revision_experiments.py \
  --config configs/revision/3d/generated.yaml \
  --tasks pseudo runtime \
  --device cuda

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)"                            | tee -a "$LOG_FILE"
echo "Primary: results/revision_pipeline_3d/summary" | tee -a "$LOG_FILE"
echo "Pseudo:  results/revision_pipeline_3d/posthoc/pseudo_label_quality.csv" | tee -a "$LOG_FILE"
echo "Runtime: results/revision_pipeline_3d/posthoc/runtime_summary.csv"      | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
