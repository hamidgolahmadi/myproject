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

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject
mkdir -p logs

echo "===== STEP 1: Generate extreme topologies ====="

python 01_generate_extreme_topologies_v2.py \
    --N 100 \
    --K 6 \
    --beta_small_world 0.02 \
    --seed 0

echo "===== STEP 2: Run baseline (no policy) ====="

python 02_run_baseline_no_policy_v2.py \
    --N 100 \
    --K 6 \
    --beta_small_world 0.02 \
    --seed 0

echo "===== PIPELINE FINISHED ====="
