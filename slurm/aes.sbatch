#!/usr/bin/env bash
#SBATCH --job-name=ian-aes
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-8
# CPU is enough: AES fits small routers on frozen expert predictions. One array task = (split seed, seed).
# usage: sbatch slurm/aes.sbatch [task] [extra hydra overrides...]
TASK="${1:-mortality24}"; shift || true
IAN_JOB_SCRIPT="aes.sbatch"; IAN_JOB_ARGS=("$TASK" "$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"
i="${SLURM_ARRAY_TASK_ID:-0}"
run_resumable ian-aes task="$TASK" split_seed=$(( i / 3 )) seed=$(( i % 3 )) "$@"
