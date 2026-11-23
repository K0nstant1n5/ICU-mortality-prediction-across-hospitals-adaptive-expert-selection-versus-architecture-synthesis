#!/usr/bin/env bash
#SBATCH --job-name=ian-nas
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@300
#SBATCH --requeue
# NAS on one split seed (SPLIT, default 0). The controller saves state after every candidate and exits 75 before the time limit;
# run_resumable resubmits until the budget is used up.
# usage: STRATEGY=evolution PRIOR=uniform INHERIT=true NAS_MODE=search SPLIT=0 sbatch slurm/nas.sbatch [task] [overrides...]
#        NAS_MODE=finalize sbatch slurm/nas.sbatch           (top-5 x 3 seeds retrain; resumable too)
TASK="${1:-mortality24}"; shift || true
IAN_JOB_SCRIPT="nas.sbatch"; IAN_JOB_ARGS=("$TASK" "$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"
run_resumable ian-nas task="$TASK" split_seed="${SPLIT:-0}" \
  nas.strategy="${STRATEGY:-evolution}" nas.prior="${PRIOR:-uniform}" nas.inherit="${INHERIT:-true}" \
  cli.nas_mode="${NAS_MODE:-search}" "$@"
