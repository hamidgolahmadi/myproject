#!/bin/bash
#SBATCH --job-name=oat_gamma
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-599
#SBATCH --output=logs/oat_gamma_%A_%a.out
#SBATCH --error=logs/oat_gamma_%A_%a.err

set -eo pipefail

mkdir -p logs
mkdir -p oat_results

source /home/hg2e25/venvs/netppo/bin/activate

TOPOLOGIES=("random" "small_world" "scale_free")

CHUNK=$((SLURM_ARRAY_TASK_ID % 200))
TOPO_INDEX=$((SLURM_ARRAY_TASK_ID / 200))
TOPO=${TOPOLOGIES[$TOPO_INDEX]}

echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "CHUNK               = ${CHUNK}"
echo "TOPOLOGY            = ${TOPO}"

python run_oat_topology_chunk.py \
  --samples_csv oat_parameter_samples/gamma_samples.csv \
  --outdir oat_results \
  --chunk_index ${CHUNK} \
  --n_chunks 200 \
  --topology ${TOPO} \
  --n_seeds 100 \
  --horizon 3000 \
  --n_workers 4
