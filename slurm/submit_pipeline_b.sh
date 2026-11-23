#!/usr/bin/env bash
# Submit the whole Pipeline B chain with SLURM dependencies. Run from the repo root after cohorts exist
# (slurm/extract_cohorts.sbatch per source) and after ./setup_env.sh.
#   bash slurm/submit_pipeline_b.sh [task]            e.g. mortality24 (default) or aki
# NAS and soup run on every split in NAS_SPLITS (default "0 1 2"; was split 0 only before 2026-11-23).
# Set SBATCH_PARTITION / SBATCH_ACCOUNT in your environment if your cluster needs them (sbatch reads both).
# Extra hydra overrides for every job: IAN_OVERRIDES="data.sources.hirid.enabled=true" bash slurm/submit_pipeline_b.sh
# DRY_RUN=1 prints the sbatch commands without submitting.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/_submit_lib.sh
TASK="${1:-mortality24}"
read -r -a OV <<< "${IAN_OVERRIDES:-}"
read -r -a NAS_SPLIT_LIST <<< "${NAS_SPLITS:-0 1 2}"
mkdir -p logs

prep=$(sub slurm/prepare.sbatch "$TASK" "${OV[@]}")
experts=$(sub $(after "$prep") slurm/train_experts.sbatch "$TASK" "${OV[@]}")
aes=$(sub $(after "$experts") slurm/aes.sbatch "$TASK" "${OV[@]}")
soup_ids=(); nas_ids=()
for split in "${NAS_SPLIT_LIST[@]}"; do
  soup_ids+=("$(submit_soup "$TASK" "$split" "$prep" "${OV[@]}")")
  read -r -a ids <<< "$(submit_nas "$TASK" "$split" "$experts" "$aes" "${OV[@]}")"
  nas_ids+=("${ids[@]}")
done
ev=$(sub $(after "$aes" "${soup_ids[@]}" "${nas_ids[@]}") slurm/evaluate.sbatch "$TASK" "${OV[@]}")

cat <<MSG
Submitted Pipeline B for task=$TASK (NAS + soup on splits ${NAS_SPLIT_LIST[*]})
  prepare      $prep
  experts      $experts   (array: split x model x seed)
  aes          $aes
  soup         ${soup_ids[*]}   (combine jobs)
  nas          ${nas_ids[*]}   (finalize jobs; searches run before them)
  evaluate     $ev
Watch: squeue -u \$USER ; logs in logs/
MSG
