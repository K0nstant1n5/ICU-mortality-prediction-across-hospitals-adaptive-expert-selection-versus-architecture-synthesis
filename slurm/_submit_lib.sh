# Shared helpers for the submit_*.sh scripts. Source it; do not run it.
# DRY_RUN=1 prints every sbatch command with fake job ids instead of submitting.

sub() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "sbatch $*" >&2; echo "DRY$RANDOM"   # fake id (subshells cannot share a counter)
  else
    sbatch --parsable "$@"
  fi
}

# after ID [ID...] -> "--dependency=afterok:ID:ID" ("-" or empty ids are skipped; prints nothing if none are left)
after() {
  local ids=() i
  for i in "$@"; do [[ -n "$i" && "$i" != "-" ]] && ids+=("$i"); done
  (( ${#ids[@]} )) && { local IFS=:; echo "--dependency=afterok:${ids[*]}"; }
  return 0
}

# Four NAS variants on one split: main run, random-search baseline (D29), D31 ablations.
# usage: submit_nas TASK SPLIT EXPERTS_JOB AES_JOB [overrides...]   ("-" = already finished, no dependency)
# prints the 4 finalize job ids
submit_nas() {
  local task=$1 split=$2 experts=$3 aes=$4; shift 4
  local ids=() variant strat prior inh s f dep
  for variant in "evolution uniform true" "random uniform true" "evolution uniform false" "evolution aes true"; do
    read -r strat prior inh <<< "$variant"
    if [[ $prior == aes ]]; then dep=$(after "$experts" "$aes"); else dep=$(after "$experts"); fi
    s=$(sub $dep --export=ALL,STRATEGY=$strat,PRIOR=$prior,INHERIT=$inh,NAS_MODE=search,SPLIT=$split \
          slurm/nas.sbatch "$task" "$@")
    f=$(sub $(after "$s") --export=ALL,STRATEGY=$strat,PRIOR=$prior,INHERIT=$inh,NAS_MODE=finalize,SPLIT=$split \
          slurm/nas.sbatch "$task" "$@")
    ids+=("$f")
  done
  echo "${ids[*]}"
}

# Model soup on one split: pretrain -> linear probe -> 10 ingredients -> combine.
# usage: submit_soup TASK SPLIT PREPARE_JOB [overrides...]   ("-" = data already prepared)
# prints the combine job id
submit_soup() {
  local task=$1 split=$2 dep=$3; shift 3
  local pre lp ing
  pre=$(sub $(after "$dep") slurm/soup.sbatch pretrain "$task" "$split" "$@")
  lp=$(sub $(after "$pre") slurm/soup.sbatch linear_probe "$task" "$split" "$@")
  ing=$(sub $(after "$lp") --array=0-9 slurm/soup.sbatch ingredient "$task" "$split" "$@")
  sub $(after "$ing") slurm/soup.sbatch combine "$task" "$split" "$@"
}
