#!/bin/bash
#SBATCH --job-name=extreme_netppo
#SBATCH --output=logs/extreme_pipeline_%j.out
#SBATCH --error=logs/extreme_pipeline_%j.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject
mkdir -p logs

echo "PWD=$(pwd)"
echo "Python=$(which python)"

echo "===== STEP 1: Generate extreme topologies ====="
python 01_generate_extreme_topologies_v2.py \
    --N 100 \
    --K 6 \
    --beta_small_world 0.02 \
    --seed 0 \
    --outdir extreme_topologies_v2

echo "===== STEP 2: Run baseline (no policy) ====="
python 02_run_baseline_no_policy_v2.py \
    --topology_dir extreme_topologies_v2 \
    --outdir baseline_results_v2 \
    --horizon 1000 \
    --seed_start 0 \
    --n_seeds 20 \
    --riskS_mean_max 8.0 \
    --ret_vol_max 0.02 \
    --belief_var_max 0.05 \
    --flow_std_max 0.20 \
    --window 50 \
    --required_consecutive_windows 3

echo "===== PIPELINE FINISHED ====="
