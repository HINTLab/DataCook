#!/usr/bin/env bash
# Tabular DataCook budget ablation with the legacy optimizer parameters.
# Project: DataCook
# Created: 2026-10-22
# Commit:  no-git
#
# Foreground run inside an existing screen session:
#   CUDA_VISIBLE_DEVICES=0 CONDA_ENV=datacook \
#     bash jobs/datacook-tabular-legacy-params-local.sh

set -euo pipefail

JOB_NAME="datacook-tabular-legacy-params"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"
DIABETES_DATA="${DIABETES_DATA:-$PROJECT_ROOT/TableData/dataset/diabetes.csv}"
HEPATITIS_DATA="${HEPATITIS_DATA:-$PROJECT_ROOT/TableData/dataset/HepatitisCdata.csv}"
SYSTEMIC_DATA="${SYSTEMIC_DATA:-$PROJECT_ROOT/TableData/dataset/systemic diseases.csv}"
TABULAR_DEVICE="${TABULAR_DEVICE:-cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GIT_COMMIT="no-git"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}-$(date +%Y%m%d-%H%M%S).log"

echo "=========================================" | tee "$LOG_FILE"
echo "Job:    $JOB_NAME"                       | tee -a "$LOG_FILE"
echo "Start:  $(date)"                         | tee -a "$LOG_FILE"
echo "Commit: $GIT_COMMIT"                     | tee -a "$LOG_FILE"
echo "GPU:    $CUDA_VISIBLE_DEVICES"           | tee -a "$LOG_FILE"
echo "Device: $TABULAR_DEVICE"                 | tee -a "$LOG_FILE"
echo "Params: epsilon=[0.1,0.25,0.5,1,2,5], step_size=0.05, num_steps=300, batch_size=32" | tee -a "$LOG_FILE"
echo "Training: Adam, lr=0.001, unweighted CE/BCEWithLogits" | tee -a "$LOG_FILE"
echo "Log:    $LOG_FILE"                       | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

for data_path in "$DIABETES_DATA" "$HEPATITIS_DATA" "$SYSTEMIC_DATA"; do
  if [[ ! -f "$data_path" ]]; then
    echo "Missing tabular CSV: $data_path" | tee -a "$LOG_FILE" >&2
    exit 2
  fi
done

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "conda command not found; using the active Python environment." | tee -a "$LOG_FILE"
fi

cd "$PROJECT_ROOT"

run_logged() {
  echo "Running: $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

run_logged python run_tabular_epsilon_ablation.py \
  --config configs/revision/tabular_legacy_params_diabetes.yaml \
  --data-path "$DIABETES_DATA" --device "$TABULAR_DEVICE"

run_logged python run_tabular_epsilon_ablation.py \
  --config configs/revision/tabular_legacy_params_hepatitis_c.yaml \
  --data-path "$HEPATITIS_DATA" --device "$TABULAR_DEVICE"

run_logged python run_tabular_epsilon_ablation.py \
  --config configs/revision/tabular_legacy_params_systemic_diseases.yaml \
  --data-path "$SYSTEMIC_DATA" --device "$TABULAR_DEVICE"

run_logged python -m analysis.merge_result_shards \
  --config configs/revision/merge_tabular_legacy_params.yaml

run_logged python -m analysis.aggregate_results \
  --config configs/revision/statistics_tabular_legacy_params.yaml

echo "-----------------------------------------" | tee -a "$LOG_FILE"
echo "End: $(date)"                            | tee -a "$LOG_FILE"
echo "Runs: results/revision_pipeline_tabular_legacy_params (54 runs)" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
