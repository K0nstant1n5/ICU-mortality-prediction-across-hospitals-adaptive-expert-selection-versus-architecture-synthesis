#!/usr/bin/env bash
#SBATCH --job-name=ian-download
#SBATCH --output=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.out
#SBATCH --error=/home/hpc/users/ml_models/konstantin.lozhankoski/dip/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
# CPU only, no GPU. Downloads MIMIC-IV v2.2 / eICU v2.0 / HiRID v1.1.1 from PhysioNet (needs internet on the compute node).
# Needs your login in ~/.netrc (a job can't ask for a password); see scripts/download_physionet.sh.
# Resumes where it stopped, so just resubmit if it hits the time limit.
# usage: sbatch slurm/download_physionet.sbatch [miiv] [eicu] [hirid]   e.g. ... download_physionet.sbatch hirid
export IAN_ROOT="${IAN_ROOT:-/home/hpc/users/ml_models/konstantin.lozhankoski/dip}"
cd "$IAN_ROOT"
mkdir -p logs
bash scripts/download_physionet.sh "$@"
