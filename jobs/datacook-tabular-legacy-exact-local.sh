#!/usr/bin/env bash
# Strict reconstruction of the three preliminary tabular DataCook results.
# The historical scripts declared a CUDA device but never moved their models or
# tensors to it, so exact reconstruction must execute on CPU.
# Run in the foreground inside screen:
#   CONDA_ENV=datacook bash jobs/datacook-tabular-legacy-exact-local.sh

set -euo pipefail

JOB_NAME="datacook-tabular-legacy-exact"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$JOB_NAME"
CONDA_ENV="${CONDA_ENV:-datacook}"

mkdir -p "$LOG_DIR"
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
echo "Protocol: strict historical reconstruction; CPU, split seed=42, epsilon=5" | tee -a "$MASTER_LOG"

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
    echo "Expected $expected_hash, found $actual_hash" | tee -a "$MASTER_LOG" >&2
    exit 2
  fi
  echo "$actual_hash  $checkpoint" | tee -a "$MASTER_LOG"
}
verify_checkpoint best_model_diabete.pt f731a062af185a5c74836088467100ffa1191998d11a5c76f687b30615956a1c
verify_checkpoint best_model_HepatitisCdata.pt 09f0ae5b47ad285c87e8a798f0c5ebb568b2463593736f16ae1eb72a4c365e7e
verify_checkpoint best_model.pt 04dd310159eaf16815e8e277fb2c75eff347b8f62b783d37e43e5aaa9e133d66

run_one() {
  local name="$1"
  local config="$2"
  local log="$LOG_DIR/$name.log"
  echo "Starting $name on CPU" | tee -a "$MASTER_LOG"
  python -u run_tabular_epsilon_ablation.py \
    --config "$config" --device cpu --force 2>&1 | tee "$log"
  echo "Completed $name" | tee -a "$MASTER_LOG"
}

# Sequential execution avoids CPU scheduling differences and mirrors the old runs.
run_one diabetes configs/revision/tabular_legacy_exact_diabetes.yaml
run_one hepatitis_c configs/revision/tabular_legacy_exact_hepatitis_c.yaml
run_one systemic_diseases configs/revision/tabular_legacy_exact_systemic_diseases.yaml

python summarize_tabular_legacy_reproduction.py --strict 2>&1 | tee -a "$MASTER_LOG"
echo "Strict legacy reconstruction completed." | tee -a "$MASTER_LOG"
