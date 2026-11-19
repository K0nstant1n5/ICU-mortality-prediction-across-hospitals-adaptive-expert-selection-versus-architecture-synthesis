#!/usr/bin/env bash
#SBATCH --job-name=ian-extract
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
# CPU only. ricu imports one source per job to stay under 8 GB; MIMIC-IV import is the heaviest step.
# usage: sbatch slurm/extract_cohorts.sbatch miiv        (then eicu, hirid when enabled)
SRC="${1:?source: miiv | eicu | hirid}"
TASKS="${2:-mortality24 aki}"
IAN_JOB_SCRIPT="extract_cohorts.sbatch"; IAN_JOB_ARGS=("$@")
source "${IAN_ROOT:-${SLURM_SUBMIT_DIR:-.}}/slurm/_common.sh"   # submit from the repo root
bash scripts/extract_cohorts.sh "$SRC" "$TASKS"

