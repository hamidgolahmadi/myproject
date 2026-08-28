#!/bin/bash
#SBATCH --job-name=oat_beta
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/oat_beta_%A_%a.out
#SBATCH --error=logs/oat_beta_%A_%a.err

set -eo pipefail

mkdir -p logs
mkdir -p oat_results

source /home/hg2e25/venvs/netppo/bin/activate

python run_oat_topology_chunk.py \
  --samples_csv oat_parameter_samples/beta_samples.csv \
  --outdir oat_results \
  --chunk_index ${SLURM_ARRAY_TASK_ID} \
  --n_chunks 200 \
  --n_seeds 100 \
  --horizon 3000
