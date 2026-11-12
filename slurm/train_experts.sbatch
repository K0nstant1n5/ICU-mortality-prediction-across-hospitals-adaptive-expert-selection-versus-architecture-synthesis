#!/usr/bin/env bash
#SBATCH --job-name=ian-expert
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --array=0-44%8
# One array task = one (split seed, model, training seed). 3 split seeds x 5 models x 3 seeds = 45 tasks.
# LightGBM ignores the GPU; submit it separately without --gres if your cluster charges for idle GPUs.
# usage: sbatch slurm/train_experts.sbatch [task] [extra hydra overrides...]
#        MODELS="gru tcn" sbatch --array=0-17 slurm/train_experts.sbatch      (subset)
TASK="${1:-mortality24}"; shift || true
IAN_JOB_SCRIPT="train_experts.sbatch"; IAN_JOB_ARGS=("$TASK" "$@")
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"   # absolute: SLURM's CWD is not reliable on this cluster
source "$IAN_ROOT/slurm/_common.sh"

read -r -a MODEL_LIST <<< "${MODELS:-lgbm grud tcn transformer mamba}"
SEEDS=(0 1 2); SPLITS=(0 1 2)
i="${SLURM_ARRAY_TASK_ID:-0}"
n_m=${#MODEL_LIST[@]}; n_s=${#SEEDS[@]}
split=${SPLITS[$(( i / (n_m * n_s) ))]}
model=${MODEL_LIST[$(( (i / n_s) % n_m ))]}
seed=${SEEDS[$(( i % n_s ))]}
echo "split_seed=$split model=$model seed=$seed"
run_resumable ian-train task="$TASK" model="$model" seed="$seed" split_seed="$split" "$@"
