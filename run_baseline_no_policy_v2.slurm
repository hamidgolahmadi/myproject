#!/bin/bash
#SBATCH --job-name=baseline_v2
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_v2_%j.out
#SBATCH --error=logs/baseline_v2_%j.err

set -euo pipefail

mkdir -p logs
mkdir -p baseline_no_policy_v2

echo "==============================================================================="
echo "Job started on $(date)"
echo "Job ID          : $SLURM_JOB_ID"
echo "Job name        : $SLURM_JOB_NAME"
echo "WorkDir         : $(pwd)"
echo "Host            : $(hostname)"
echo "==============================================================================="

source ~/.bashrc
conda activate netppo

python 02_run_baseline_no_policy_v2.py \
  --topology_dir topologies_extreme \
  --outdir baseline_no_policy_v2 \
  --horizon 3000 \
  --seed_start 0 \
  --n_seeds 10 \
  --riskS_mean_max 2.0 \
  --ret_vol_max 0.0020 \
  --belief_var_max 0.015 \
  --flow_std_max 0.12 \
  --window 50 \
  --required_consecutive_windows 3

echo "==============================================================================="
echo "Job finished on $(date)"
echo "==============================================================================="
