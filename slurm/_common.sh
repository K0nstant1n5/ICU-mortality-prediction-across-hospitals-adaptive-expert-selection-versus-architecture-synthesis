#!/usr/bin/env bash
# Sourced by every job script. Sets paths, activates the venv, and provides `run_resumable`.
#
# Resubmission contract: a Python entry point exits with 75 when it stopped early to stay inside the
# 8 h job limit (state is checkpointed). `run_resumable` then re-queues the *same* job script with the
# same environment, so the next allocation continues from the checkpoint. Any other non-zero code fails the job.
set -euo pipefail

: "${IAN_ROOT:=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export IAN_ROOT
export IAN_DATA="${IAN_DATA:-$IAN_ROOT/data}"
export IAN_ARTIFACTS="${IAN_ARTIFACTS:-$IAN_ROOT/artifacts}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# keep CPU thread pools inside the allocation
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

cd "$IAN_ROOT"
mkdir -p "$IAN_ROOT/logs"
if [[ -f "$IAN_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$IAN_ROOT/.venv/bin/activate"
else
  echo "No .venv found in $IAN_ROOT; run ./setup_env.sh first" >&2
  exit 1
fi

MAX_RESUBMITS="${MAX_RESUBMITS:-20}"
RESTARTS="${SLURM_RESTART_COUNT:-0}"

echo "[$(date -Is)] job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-} node=$(hostname) restart=${RESTARTS}"
python - <<'PY' 2>/dev/null || true
import torch
try:  # on CPU-only jobs the node's driver is visible but no GPU is assigned; don't print a traceback for that
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else ''
except Exception:
    name = ''
print('torch', torch.__version__, 'cuda', bool(name), name)
PY
gpu_q() { local v; v=$(nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null | head -1) || true; echo "${v:-N/A}"; }
echo "GPU: $(gpu_q name)  VRAM: $(gpu_q memory.total)"

# run_resumable <command...>: run; on exit 75 put this same job back in the queue.
# `scontrol requeue` keeps the job id, so `--dependency=afterok:<id>` in submit_pipeline_b.sh only fires
# once the work is really finished. If your cluster forbids user requeues, it falls back to a fresh sbatch
# (then dependent jobs must be resubmitted by hand; the log says so).
run_resumable() {
  set +e
  "$@"
  local rc=$?
  set -e
  [[ $rc -ne 75 ]] && return $rc
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Incomplete (exit 75) outside SLURM: run the same command again to continue." >&2
    return 75
  fi
  if (( RESTARTS >= MAX_RESUBMITS )); then
    echo "Reached MAX_RESUBMITS=$MAX_RESUBMITS; giving up" >&2
    return 1
  fi
  local id="${SLURM_JOB_ID}"
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  echo "Incomplete: requeueing $id"
  if scontrol requeue "$id"; then
    return 0
  fi
  echo "WARNING: scontrol requeue refused; submitting a new job. Jobs that depended on $id will not wait for it." >&2
  local script="$IAN_ROOT/slurm/${IAN_JOB_SCRIPT:?}"
  local array_opt=()
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && array_opt=(--array="${SLURM_ARRAY_TASK_ID}")
  sbatch --export=ALL "${array_opt[@]}" "$script" "${IAN_JOB_ARGS[@]}"
}
