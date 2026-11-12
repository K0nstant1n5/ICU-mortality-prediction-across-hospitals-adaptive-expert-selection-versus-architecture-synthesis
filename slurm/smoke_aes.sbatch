#!/usr/bin/env bash
#SBATCH --job-name=ian-smoke-aes
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Smoke test on fabricated data (no real patient data). Runs on a GPU to also exercise CUDA.
# usage: sbatch slurm/smoke_aes.sbatch
IAN_JOB_SCRIPT="smoke_aes.sbatch"; IAN_JOB_ARGS=("$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"
python -m pytest -m smoke_aes tests/smoke -q -x "$@"
