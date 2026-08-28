#!/bin/bash
#SBATCH --job-name=base1000v2
#SBATCH --output=logs/base1000v2_%A_%a.out
#SBATCH --error=logs/base1000v2_%A_%a.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-99

set -euo pipefail

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject
mkdir -p logs
mkdir -p baseline_no_policy_1000seeds_v2

SEEDS_PER_JOB=10
SEED_START=$((SLURM_ARRAY_TASK_ID * SEEDS_PER_JOB))
JOB_OUTDIR="baseline_no_policy_1000seeds_v2/chunk_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$JOB_OUTDIR"

echo "==============================================================================="
echo "Job started on $(date)"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "SEED_START          = $SEED_START"
echo "Running seeds       = $SEED_START to $((SEED_START + SEEDS_PER_JOB - 1))"
echo "JOB_OUTDIR          = $JOB_OUTDIR"
echo "==============================================================================="

python 02_run_baseline_no_policy_v2.py \
  --topology_dir extreme_topologies_v2 \
  --outdir "$JOB_OUTDIR" \
  --horizon 1000 \
  --seed_start "$SEED_START" \
  --n_seeds "$SEEDS_PER_JOB" \
  --riskS_mean_max 8.0 \
  --ret_vol_max 0.02 \
  --belief_var_max 0.05 \
  --flow_std_max 0.20 \
  --window 50 \
  --required_consecutive_windows 3

echo "==============================================================================="
echo "Job finished on $(date)"
echo "==============================================================================="
