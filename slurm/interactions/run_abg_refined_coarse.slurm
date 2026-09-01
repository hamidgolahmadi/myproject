#!/bin/bash
#SBATCH --job-name=abg_coarse
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-299
#SBATCH --output=logs/abg_coarse_%A_%a.out
#SBATCH --error=logs/abg_coarse_%A_%a.err

set -eo pipefail

mkdir -p logs
mkdir -p interaction_results_coarse

source .venv/bin/activate
unset PYTHONPATH

TOPOLOGIES=("random" "small_world" "scale_free")

CHUNK=$((SLURM_ARRAY_TASK_ID % 100))
TOPO_INDEX=$((SLURM_ARRAY_TASK_ID / 100))
TOPO=${TOPOLOGIES[$TOPO_INDEX]}

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "CHUNK               = ${CHUNK}"
echo "TOPOLOGY            = ${TOPO}"

python run_interaction_topology_chunk.py \
  --samples_csv interaction_parameter_samples/abg_alpha125_250_coarse.csv \
  --outdir interaction_results_coarse \
  --chunk_index ${CHUNK} \
  --n_chunks 100 \
  --topology ${TOPO} \
  --n_seeds 50 \
  --horizon 3000 \
  --n_workers 4
