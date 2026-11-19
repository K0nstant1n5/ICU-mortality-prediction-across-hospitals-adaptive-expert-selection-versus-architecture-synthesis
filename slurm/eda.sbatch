#!/usr/bin/env bash
#SBATCH --job-name=ian-eda
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
# CPU only. Executes notebooks/eda.ipynb headless. Figures/tables go to outputs/eda/ (or $EDA_OUT).
# usage: sbatch slurm/eda.sbatch
#        EDA_SOURCES=miiv,eicu,hirid EDA_TASKS=mortality24 sbatch slurm/eda.sbatch   # optional overrides
source "${IAN_ROOT:-${SLURM_SUBMIT_DIR:-.}}/slurm/_common.sh"   # submit from the repo root
export EDA_OUT="${EDA_OUT:-$IAN_ROOT/outputs/eda}"
mkdir -p "$EDA_OUT"
jupyter nbconvert --to notebook --execute notebooks/eda.ipynb \
  --output-dir "$EDA_OUT" --output eda_executed.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to html "$EDA_OUT/eda_executed.ipynb" --output-dir "$EDA_OUT"
echo "Done: $EDA_OUT/eda_executed.html, figures in $EDA_OUT/figures, tables in $EDA_OUT/tables"
