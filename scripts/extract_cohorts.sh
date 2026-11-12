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

if [[ -x "$ROOT/.r-env/bin/micromamba" ]]; then
  eval "$("$ROOT/.r-env/bin/micromamba" shell hook -s bash)"
  micromamba activate "$ROOT/.r-env/env"
fi
command -v Rscript >/dev/null || { echo "Rscript not found. Run ./setup_env.sh --with-r first."; exit 1; }
[[ -d "$COHORTS_REPO" ]] || { echo "Missing $COHORTS_REPO. Run ./setup_env.sh first."; exit 1; }

# 1) raw CSV -> fst (skips tables that are already imported)
Rscript "$ROOT/r/import_ricu.R" "$DATA/raw" $SOURCES

# 2) YAIB-cohorts reads ../config.yaml relative to its R/ folder
cat > "$COHORTS_REPO/config.yaml" <<YAML
ricu_data_path: "$DATA/raw"
out_dir: "$DATA/cohorts"
YAML

declare -A SCRIPT=( [mortality24]=mortality.R [aki]=aki.R )
cd "$COHORTS_REPO/R"
for src in $SOURCES; do
  echo "== base cohort: $src"
  Rscript base_cohort.R --src "$src"
  for task in $TASKS; do
    echo "== $task: $src"
    Rscript "${SCRIPT[$task]}" --src "$src"
  done
done
echo "Cohorts written to $DATA/cohorts"
