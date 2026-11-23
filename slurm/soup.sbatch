#!/usr/bin/env bash
#SBATCH --job-name=ian-soup
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@300
#SBATCH --requeue
# Model soup in four stages (submit_pipeline_b.sh chains them):
#   sbatch slurm/soup.sbatch pretrain  [task] [split]
#   sbatch slurm/soup.sbatch linear_probe [task] [split]                (head-only fit = shared init, LP-FT)
#   sbatch --array=0-9 slurm/soup.sbatch ingredient [task] [split]     (one fine-tune per array task)
#   sbatch slurm/soup.sbatch combine   [task] [split]
STAGE="${1:?pretrain | linear_probe | ingredient | combine}"
TASK="${2:-mortality24}"; SPLIT="${3:-0}"; shift $(( $# < 3 ? $# : 3 ))
IAN_JOB_SCRIPT="soup.sbatch"; IAN_JOB_ARGS=("$STAGE" "$TASK" "$SPLIT" "$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"
run_resumable ian-pretrain-soup task="$TASK" split_seed="$SPLIT" cli.soup_stage="$STAGE" \
  cli.ingredient="${SLURM_ARRAY_TASK_ID:-0}" "$@"
