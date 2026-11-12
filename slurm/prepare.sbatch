#!/usr/bin/env bash
#SBATCH --job-name=ian-prepare
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
# CPU only. Splits (3 split seeds), fits the preprocessor on pooled train, writes memory-mapped arrays.
# usage: sbatch slurm/prepare.sbatch [task] [extra hydra overrides...]
TASK="${1:-mortality24}"; shift || true
IAN_JOB_SCRIPT="prepare.sbatch"; IAN_JOB_ARGS=("$TASK" "$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"
run_resumable ian-prepare task="$TASK" "$@"
