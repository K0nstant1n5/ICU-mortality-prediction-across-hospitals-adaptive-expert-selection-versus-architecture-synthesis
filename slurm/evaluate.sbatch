#!/usr/bin/env bash
#SBATCH --job-name=ian-eval
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-8
# Metrics, bootstrap CIs (clustered by hospital on eICU), DeLong + paired bootstrap with Holm. One array task = (split seed, training seed).
# usage: sbatch slurm/evaluate.sbatch [task] [overrides...]
TASK="${1:-mortality24}"; shift || true
IAN_JOB_SCRIPT="evaluate.sbatch"; IAN_JOB_ARGS=("$TASK" "$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"
run_resumable ian-evaluate task="$TASK" split_seed=$(( ${SLURM_ARRAY_TASK_ID:-0} / 3 )) seed=$(( ${SLURM_ARRAY_TASK_ID:-0} % 3 )) "$@"
