#!/usr/bin/env bash
# Runs ricu import + YAIB-cohorts for the enabled sources and tasks.
# Usage: scripts/extract_cohorts.sh "miiv eicu" "mortality24 aki"
#   (add hirid / aumc to the first argument once downloaded)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${IAN_DATA:-$ROOT/data}"
SOURCES="${1:-miiv eicu}"
TASKS="${2:-mortality24 aki}"
COHORTS_REPO="$ROOT/third_party/YAIB-cohorts"

# Use the R from ./setup_env.sh --with-r if present. Putting its bin/ first on PATH is all "activate" does for
# Rscript, and it avoids micromamba's shell hook (which breaks under `set -u`).
if [[ -x "$ROOT/.r-env/env/bin/Rscript" ]]; then
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT/.r-env/root}"
  export PATH="$ROOT/.r-env/env/bin:$PATH"
  export CONDA_PREFIX="$ROOT/.r-env/env"
fi
command -v Rscript >/dev/null || { echo "Rscript not found. Run ./setup_env.sh --with-r first."; exit 1; }
[[ -d "$COHORTS_REPO" ]] || { echo "Missing $COHORTS_REPO. Run ./setup_env.sh first."; exit 1; }

# 0) ricu's miiv config looks for hosp/caregiver.csv.gz, but MIMIC-IV v2.2 ships it in icu/. Without it ricu
#    sees "30 of 31 tables" and refuses to attach miiv. A relative symlink fixes this without copying data.
if [[ " $SOURCES " == *" miiv "* && -f "$DATA/raw/miiv/icu/caregiver.csv.gz" && ! -e "$DATA/raw/miiv/hosp/caregiver.csv.gz" ]]; then
  ln -s ../icu/caregiver.csv.gz "$DATA/raw/miiv/hosp/caregiver.csv.gz"
  echo "Linked miiv hosp/caregiver.csv.gz -> icu/caregiver.csv.gz"
fi

# 1) raw CSV -> fst (skips tables that are already imported)
Rscript "$ROOT/r/import_ricu.R" "$DATA/raw" $SOURCES

# 2) YAIB-cohorts reads ../config.yaml relative to its R/ folder
cat > "$COHORTS_REPO/config.yaml" <<YAML
ricu_data_path: "$DATA/raw"
out_dir: "$DATA/cohorts"
YAML

# YAIB-cohorts' R/.Rprofile switches on renv, which hides the packages installed by setup_env.sh.
# Skip it: the same versions (ricu fork, argparser, ...) are already installed in .r-env.
export RENV_ACTIVATE_PROJECT=FALSE
RS=(Rscript --no-init-file)
declare -A SCRIPT=( [mortality24]=mortality.R [aki]=aki.R )
cd "$COHORTS_REPO/R"
for src in $SOURCES; do
  echo "== base cohort: $src"
  "${RS[@]}" base_cohort.R --src "$src"
  for task in $TASKS; do
    echo "== $task: $src"
    "${RS[@]}" "${SCRIPT[$task]}" --src "$src"
  done
done
echo "Cohorts written to $DATA/cohorts"
