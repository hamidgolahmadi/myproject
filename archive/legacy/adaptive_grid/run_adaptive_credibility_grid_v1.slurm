#!/bin/bash
#SBATCH --job-name=acred
#SBATCH --output=logs/acred_%A_%a.out
#SBATCH --error=logs/acred_%A_%a.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-9

set -euo pipefail

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject
mkdir -p logs
mkdir -p adaptive_credibility_grid_v1

# ----------------------------------------
# User-set parameters via environment vars
# Must be exported before sbatch:
#   TOPOLOGY
#   BETA
#   GAMMA
# ----------------------------------------

: "${TOPOLOGY:?Need TOPOLOGY}"
: "${BETA:?Need BETA}"
: "${GAMMA:?Need GAMMA}"

SEEDS_PER_JOB=20
SEED_START=$((SLURM_ARRAY_TASK_ID * SEEDS_PER_JOB))

python 03_run_adaptive_credibility_grid_v1.py \
  --topology_dir extreme_topologies_v2 \
  --outdir adaptive_credibility_grid_v1 \
  --topology "${TOPOLOGY}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --seed_start "${SEED_START}" \
  --n_seeds "${SEEDS_PER_JOB}" \
  --horizon 1000 \
  --riskS_mean_max 8.0 \
  --ret_vol_max 0.02 \
  --belief_var_max 0.05 \
  --flow_std_max 0.20 \
  --window 50 \
  --required_consecutive_windows 3 \
  --explosion_riskS_mean 1000.0
