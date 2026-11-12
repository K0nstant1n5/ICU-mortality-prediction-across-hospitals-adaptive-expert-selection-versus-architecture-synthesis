#!/usr/bin/env bash
# Submit the whole Pipeline B chain with SLURM dependencies. Run from the repo root after cohorts exist
# (slurm/extract_cohorts.sbatch per source) and after ./setup_env.sh.
#   bash slurm/submit_pipeline_b.sh [task]            e.g. mortality24 (default) or aki
# Set SBATCH_PARTITION / SBATCH_ACCOUNT in your environment if your cluster needs them (sbatch reads both).
# Extra hydra overrides for every job: IAN_OVERRIDES="data.sources.hirid.enabled=true" bash slurm/submit_pipeline_b.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TASK="${1:-mortality24}"
read -r -a OV <<< "${IAN_OVERRIDES:-}"
mkdir -p logs
sub() { sbatch --parsable "$@"; }

prep=$(sub slurm/prepare.sbatch "$TASK" "${OV[@]}")
experts=$(sub --dependency=afterok:$prep slurm/train_experts.sbatch "$TASK" "${OV[@]}")
soup_pre=$(sub --dependency=afterok:$prep slurm/soup.sbatch pretrain "$TASK" 0 "${OV[@]}")
soup_ing=$(sub --dependency=afterok:$soup_pre --array=0-9 slurm/soup.sbatch ingredient "$TASK" 0 "${OV[@]}")
soup_comb=$(sub --dependency=afterok:$soup_ing slurm/soup.sbatch combine "$TASK" 0 "${OV[@]}")
aes=$(sub --dependency=afterok:$experts slurm/aes.sbatch "$TASK" "${OV[@]}")

# NAS (split seed 0): main run, random-search baseline (D29), and the D31 ablations.
nas_ids=()
for variant in "evolution uniform true" "random uniform true" "evolution uniform false" "evolution aes true"; do
  read -r strat prior inh <<< "$variant"
  dep="afterok:$experts"; [[ $prior == aes ]] && dep="afterok:$experts:$aes"
  s=$(STRATEGY=$strat PRIOR=$prior INHERIT=$inh NAS_MODE=search sub --dependency=$dep \
        --export=ALL,STRATEGY=$strat,PRIOR=$prior,INHERIT=$inh,NAS_MODE=search slurm/nas.sbatch "$TASK" "${OV[@]}")
  f=$(sub --dependency=afterok:$s \
        --export=ALL,STRATEGY=$strat,PRIOR=$prior,INHERIT=$inh,NAS_MODE=finalize slurm/nas.sbatch "$TASK" "${OV[@]}")
  nas_ids+=("$f")
done
deps=$(IFS=:; echo "$aes:$soup_comb:${nas_ids[*]}")
ev=$(sub --dependency=afterok:$deps slurm/evaluate.sbatch "$TASK" "${OV[@]}")

cat <<MSG
Submitted Pipeline B for task=$TASK
  prepare      $prep
  experts      $experts   (array: split x model x seed)
  soup         $soup_pre -> $soup_ing -> $soup_comb
  aes          $aes
  nas          ${nas_ids[*]}   (finalize jobs; searches run before them)
  evaluate     $ev
Watch: squeue -u \$USER ; logs in logs/
MSG
