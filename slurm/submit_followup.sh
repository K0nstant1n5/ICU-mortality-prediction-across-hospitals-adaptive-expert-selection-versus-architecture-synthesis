#!/usr/bin/env bash
# Queue the follow-up runs after the first full mortality24 run (2026-11-23), all at once with dependencies.
#   bash slurm/submit_followup.sh [part ...]        parts: splits nomask lodo aki   (default: all four)
#   DRY_RUN=1 bash slurm/submit_followup.sh         print the sbatch commands only
#
#   splits  mortality24: NAS (4 variants) + soup on split seeds 1 and 2 (split 0 is done), so NAS/soup are
#           repeated over the same 3 splits as experts and AES.
#   nomask  mortality24 missingness ablation (D12): the 4 deep experts with trainer.use_missingness_mask=false
#           (-> <model>_nomask) and LightGBM without count features (-> lgbm_nocounts), 3 splits x 3 seeds.
#           Names contain "_no", so they never become "best_expert".
#   -> then one evaluation of mortality24 (9 array tasks) after everything above, so the report includes it all.
#   lodo    leave-one-database-out (D25-28), mortality24: HiRID held out (train MIMIC-IV + eICU) and MIMIC-IV held
#           out (train eICU + HiRID); the eICU-held-out fold is the main run. Experts + AES + evaluation, in
#           separate folders data/processed_lodo_<db> and artifacts_lodo_<db> (own cost ledger), so the main run
#           is never touched. Includes the main NAS variant (evolution, uniform, inherit) on split 0 of each fold; LODO_NAS=0 skips it.
#   aki     whole Pipeline B for the AKI task (slurm/submit_pipeline_b.sh aki; NAS + soup on all 3 splits).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
source slurm/_submit_lib.sh
read -r -a OV <<< "${IAN_OVERRIDES:-}"
PARTS=("$@"); (( ${#PARTS[@]} )) || PARTS=(splits nomask lodo aki)
has() { [[ " ${PARTS[*]} " == *" $1 "* ]]; }
mkdir -p logs
T=mortality24
main_deps=()

if has splits; then
  for split in 1 2; do
    c=$(submit_soup "$T" "$split" - "${OV[@]}")
    read -r -a ids <<< "$(submit_nas "$T" "$split" - - "${OV[@]}")"
    main_deps+=("$c" "${ids[@]}")
    echo "splits: split $split soup combine $c, NAS finalize ${ids[*]}"
  done
fi

if has nomask; then
  deep=$(MODELS="grud tcn transformer mamba" sub --array=0-35%8 \
           slurm/train_experts.sbatch "$T" trainer.use_missingness_mask=false "${OV[@]}")
  gbm=$(MODELS=lgbm sub --array=0-8 slurm/train_experts.sbatch "$T" model.use_count_features=false "${OV[@]}")
  main_deps+=("$deep" "$gbm")
  echo "nomask: deep experts $deep, lgbm_nocounts $gbm"
fi

if (( ${#main_deps[@]} )); then
  ev=$(sub $(after "${main_deps[@]}") slurm/evaluate.sbatch "$T" "${OV[@]}")
  echo "mortality24 evaluation (all 9 runs) $ev"
fi

if has lodo; then
  for held in hirid miiv; do
    lov=(data.sources.$held.role=test data.sources.eicu.role=train
         paths.processed="$ROOT/data/processed_lodo_$held" paths.artifacts="$ROOT/artifacts_lodo_$held" "${OV[@]}")
    p=$(sub slurm/prepare.sbatch "$T" "${lov[@]}")
    e=$(sub $(after "$p") slurm/train_experts.sbatch "$T" "${lov[@]}")
    a=$(sub $(after "$e") slurm/aes.sbatch "$T" "${lov[@]}")
    ndeps=()
    if [[ "${LODO_NAS:-1}" == 1 ]]; then
      s=$(sub $(after "$e") --export=ALL,STRATEGY=evolution,PRIOR=uniform,INHERIT=true,NAS_MODE=search,SPLIT=0 \
            slurm/nas.sbatch "$T" "${lov[@]}")
      ndeps+=("$(sub $(after "$s") --export=ALL,STRATEGY=evolution,PRIOR=uniform,INHERIT=true,NAS_MODE=finalize,SPLIT=0 \
            slurm/nas.sbatch "$T" "${lov[@]}")")
    fi
    v=$(sub $(after "$a" "${ndeps[@]}") slurm/evaluate.sbatch "$T" "${lov[@]}")
    echo "lodo (held out: $held): prepare $p, experts $e, aes $a, ${ndeps:+nas $ndeps, }evaluate $v"
  done
fi

if has aki; then
  bash slurm/submit_pipeline_b.sh aki
fi
has aki || echo "Watch: squeue -u \$USER ; logs in logs/"
