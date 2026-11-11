#!/usr/bin/env bash
# Five-GPU sharded tabular DataCook budget ablation.
# Default physical GPU IDs: 0,1,3,4,5.
#
# Run in the foreground inside screen:
#   GPU_LIST=0,1,3,4,5 CONDA_ENV=datacook \
#     bash jobs/datacook-tabular-legacy-params-5gpu-local.sh

set -euo pipefail

JOB_NAME="datacook-tabular-legacy-params-5gpu"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"
GPU_LIST="${GPU_LIST:-0,1,3,4,5}"
DIABETES_DATA="${DIABETES_DATA:-$PROJECT_ROOT/TableData/dataset/diabetes.csv}"
HEPATITIS_DATA="${HEPATITIS_DATA:-$PROJECT_ROOT/TableData/dataset/HepatitisCdata.csv}"
SYSTEMIC_DATA="${SYSTEMIC_DATA:-$PROJECT_ROOT/TableData/dataset/systemic diseases.csv}"
ARTIFACT_BASE="$PROJECT_ROOT/artifacts/revision_pipeline_tabular_legacy_params/shards"
RESULT_BASE="$PROJECT_ROOT/results/revision_pipeline_tabular_legacy_params/shards"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$ARTIFACT_BASE" "$RESULT_BASE"
MASTER_LOG="$LOG_DIR/$JOB_NAME-$(date +%Y%m%d-%H%M%S).log"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} < 5 )); then
  echo "GPU_LIST must provide at least five comma-separated GPU IDs." >&2
  exit 2
fi

for data_path in "$DIABETES_DATA" "$HEPATITIS_DATA" "$SYSTEMIC_DATA"; do
  if [[ ! -f "$data_path" ]]; then
    echo "Missing tabular CSV: $data_path" >&2
    exit 2
  fi
done

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  echo "Using active environment: $CONDA_PREFIX" | tee "$MASTER_LOG"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  echo "Activated conda environment: $CONDA_ENV" | tee "$MASTER_LOG"
else
  echo "No conda command found; using active Python: $(command -v python)" | tee "$MASTER_LOG"
fi

cd "$PROJECT_ROOT"
echo "GPU list: $GPU_LIST" | tee -a "$MASTER_LOG"
echo "Grid: 3 datasets x 3 seeds x 6 budgets = 54 runs" | tee -a "$MASTER_LOG"
echo "Parameters: step_size=0.05, num_steps=300, batch_size=32" | tee -a "$MASTER_LOG"

run_shard() {
  local gpu="$1"
  local shard="$2"
  local config="$3"
  local data_path="$4"
  shift 4
  local shard_log="$LOG_DIR/$shard.log"
  echo "Starting $shard on physical GPU $gpu; log=$shard_log" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="$gpu" python -u run_tabular_epsilon_ablation.py \
    --config "$config" \
    --data-path "$data_path" \
    --device cuda \
    --artifact-root "$ARTIFACT_BASE/$shard" \
    --result-root "$RESULT_BASE/$shard" \
    "$@" >"$shard_log" 2>&1
}

pids=()
names=()

(
  run_shard "${GPUS[0]}" systemic-s0-low \
    configs/revision/tabular_legacy_params_systemic_diseases.yaml "$SYSTEMIC_DATA" \
    --seeds 0 --epsilons 0.1 0.25 0.5 1
) &
pids+=("$!"); names+=("systemic-s0-low")

(
  run_shard "${GPUS[1]}" systemic-s0-high \
    configs/revision/tabular_legacy_params_systemic_diseases.yaml "$SYSTEMIC_DATA" \
    --seeds 0 --epsilons 2 5
  run_shard "${GPUS[1]}" systemic-s1-low \
    configs/revision/tabular_legacy_params_systemic_diseases.yaml "$SYSTEMIC_DATA" \
    --seeds 1 --epsilons 0.1 0.25
) &
pids+=("$!"); names+=("systemic-s0-high+systemic-s1-low")

(
  run_shard "${GPUS[2]}" systemic-s1-high \
    configs/revision/tabular_legacy_params_systemic_diseases.yaml "$SYSTEMIC_DATA" \
    --seeds 1 --epsilons 0.5 1 2 5
) &
pids+=("$!"); names+=("systemic-s1-high")

(
  run_shard "${GPUS[3]}" systemic-s2-low \
    configs/revision/tabular_legacy_params_systemic_diseases.yaml "$SYSTEMIC_DATA" \
    --seeds 2 --epsilons 0.1 0.25 0.5
  run_shard "${GPUS[3]}" diabetes \
    configs/revision/tabular_legacy_params_diabetes.yaml "$DIABETES_DATA"
) &
pids+=("$!"); names+=("systemic-s2-low+diabetes")

(
  run_shard "${GPUS[4]}" systemic-s2-high \
    configs/revision/tabular_legacy_params_systemic_diseases.yaml "$SYSTEMIC_DATA" \
    --seeds 2 --epsilons 1 2 5
  run_shard "${GPUS[4]}" hepatitis_c \
    configs/revision/tabular_legacy_params_hepatitis_c.yaml "$HEPATITIS_DATA"
) &
pids+=("$!"); names+=("systemic-s2-high+hepatitis_c")

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${names[$index]}" | tee -a "$MASTER_LOG"
  else
    echo "FAILED ${names[$index]}" | tee -a "$MASTER_LOG" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "At least one shard failed; merge was not run." | tee -a "$MASTER_LOG" >&2
  exit 1
fi

python -m analysis.merge_result_shards \
  --config configs/revision/merge_tabular_legacy_params_7gpu.yaml \
  2>&1 | tee -a "$MASTER_LOG"
python -m analysis.aggregate_results \
  --config configs/revision/statistics_tabular_legacy_params.yaml \
  2>&1 | tee -a "$MASTER_LOG"

echo "All 54 runs and final aggregation completed." | tee -a "$MASTER_LOG"
