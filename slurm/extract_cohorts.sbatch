#!/usr/bin/env bash
#SBATCH --job-name=ian-extract
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
# CPU only. ricu imports one source per job to keep memory low; MIMIC-IV import is the heaviest step.
# usage: sbatch slurm/extract_cohorts.sbatch miiv        (then eicu, hirid when enabled)
SRC="${1:?source: miiv | eicu | hirid}"
TASKS="${2:-mortality24 aki}"
IAN_JOB_SCRIPT="extract_cohorts.sbatch"; IAN_JOB_ARGS=("$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"
bash scripts/extract_cohorts.sh "$SRC" "$TASKS"
