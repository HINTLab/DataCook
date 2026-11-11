#!/usr/bin/env bash
# Eight-seed perturbation-budget study under the faithfully reconstructed
# historical tabular protocols. This is CPU-only because the old code executed
# its models and tensors on CPU despite declaring a CUDA device.
#
# Foreground run inside screen:
#   CONDA_ENV=datacook bash jobs/datacook-tabular-legacy-extended-local.sh

set -euo pipefail

JOB_NAME="datacook-tabular-legacy-extended"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
OUTPUT_DIR="$PROJECT_ROOT/outputs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
MASTER_LOG="$LOG_DIR/$JOB_NAME-$(date +%Y%m%d-%H%M%S).log"

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
echo "Protocol: fixed historical split/checkpoints; 8 victim seeds; 6 budgets" | tee -a "$MASTER_LOG"
echo "Grid: 3 datasets x 8 seeds x 6 budgets = 144 runs" | tee -a "$MASTER_LOG"
echo "Diabetes/HepatitisC: T=300, alpha=0.05" | tee -a "$MASTER_LOG"
echo "SystemicDiseases: T=80, alpha=epsilon/4" | tee -a "$MASTER_LOG"
python -c 'import platform, sklearn, torch; print(f"Python={platform.python_version()} PyTorch={torch.__version__} sklearn={sklearn.__version__}")' | tee -a "$MASTER_LOG"

CHECKPOINT_DIR="$PROJECT_ROOT/TableData/TableData_oldversion"
verify_checkpoint() {
  local filename="$1"
  local expected_hash="$2"
  checkpoint="$CHECKPOINT_DIR/$filename"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing historical checkpoint: $checkpoint" | tee -a "$MASTER_LOG" >&2
    exit 2
  fi
  actual_hash="$(sha256sum "$checkpoint" | awk '{print $1}')"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    echo "Checkpoint hash mismatch: $checkpoint" | tee -a "$MASTER_LOG" >&2
    exit 2
  fi
  echo "$actual_hash  $checkpoint" | tee -a "$MASTER_LOG"
}
verify_checkpoint best_model_diabete.pt f731a062af185a5c74836088467100ffa1191998d11a5c76f687b30615956a1c
verify_checkpoint best_model_HepatitisCdata.pt 09f0ae5b47ad285c87e8a798f0c5ebb568b2463593736f16ae1eb72a4c365e7e
verify_checkpoint best_model.pt 04dd310159eaf16815e8e277fb2c75eff347b8f62b783d37e43e5aaa9e133d66

run_dataset() {
  local name="$1"
  local config="$2"
  local log="$LOG_DIR/$name.log"
  echo "Starting $name on CPU; log=$log" | tee -a "$MASTER_LOG"
  python -u run_tabular_epsilon_ablation.py \
    --config "$config" --device cpu >"$log" 2>&1
}

run_dataset diabetes configs/revision/tabular_legacy_extended_diabetes.yaml &
pid_diabetes=$!
run_dataset hepatitis_c configs/revision/tabular_legacy_extended_hepatitis_c.yaml &
pid_hepatitis=$!
run_dataset systemic_diseases configs/revision/tabular_legacy_extended_systemic_diseases.yaml &
pid_systemic=$!

failed=0
for spec in "diabetes:$pid_diabetes" "hepatitis_c:$pid_hepatitis" "systemic_diseases:$pid_systemic"; do
  name="${spec%%:*}"
  pid="${spec##*:}"
  if wait "$pid"; then
    echo "Completed $name" | tee -a "$MASTER_LOG"
  else
    echo "FAILED $name; inspect $LOG_DIR/$name.log" | tee -a "$MASTER_LOG" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

python -m analysis.merge_result_shards \
  --config configs/revision/merge_tabular_legacy_extended.yaml \
  2>&1 | tee -a "$MASTER_LOG"
python -m analysis.aggregate_results \
  --config configs/revision/statistics_tabular_legacy_extended.yaml \
  2>&1 | tee -a "$MASTER_LOG"
python -m analysis.tabular_legacy_statistics \
  --config configs/revision/statistics_tabular_legacy_extended.yaml \
  2>&1 | tee -a "$MASTER_LOG"

echo "All 144 runs and final aggregation completed." | tee -a "$MASTER_LOG"
