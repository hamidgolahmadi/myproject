#!/bin/bash
#SBATCH --job-name=gen_topo_v2
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_topo_v2_%j.out
#SBATCH --error=logs/gen_topo_v2_%j.err

set -eo pipefail

mkdir -p logs
mkdir -p topologies_extreme

echo "=================================================="
echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "=================================================="

source ~/.bashrc
conda activate netppo

echo "Using shell: $SHELL"
echo "which python:  $(which python || true)"
echo "which python3: $(which python3 || true)"
echo "conda env:     ${CONDA_DEFAULT_ENV:-none}"

python3 01_generate_extreme_topologies_v2.py \
  --N 100 \
  --K 6 \
  --beta_small_world 0.02 \
  --seed 0 \
  --outdir topologies_extreme

echo "=================================================="
echo "Finished on $(date)"
echo "=================================================="
