#!/usr/bin/env bash
# Local single-GPU experiment runner generated for the 3D DataCook smoke test.
# Project: DataCook
# Created: 2026-07-21
# Commit:  1566c03404
#
# Run inside screen: bash jobs/datacook-3d-smoke-local.sh [GPU_ID]

set -euo pipefail

JOB_NAME="datacook-3d-smoke"
LOG_DIR="outputs/logs/${JOB_NAME}"
OUTPUT_DIR="results/revision_pipeline_3d"
GPU_ID="${1:-0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate datacook
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Job: $JOB_NAME" | tee "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "Command: python -u run_datacook_suite.py --config configs/pipelines/medmnist3d_smoke.yaml" | tee -a "$LOG_FILE"

set +e
python -u run_datacook_suite.py \
  --config configs/pipelines/medmnist3d_smoke.yaml 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "End: $(date)" | tee -a "$LOG_FILE"
echo "Exit: $EXIT_CODE" | tee -a "$LOG_FILE"
exit "$EXIT_CODE"
