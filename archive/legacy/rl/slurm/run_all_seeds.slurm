#!/bin/bash
#SBATCH --job-name=netppo
#SBATCH --output=logs/netppo_%A_%a.out
#SBATCH --error=logs/netppo_%A_%a.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-59

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject
mkdir -p logs

# -------------------
# topology selection
# -------------------

topologies=("random_fixed" "scale_free" "small_world")

topo_index=$((SLURM_ARRAY_TASK_ID / 20))
seed=$((SLURM_ARRAY_TASK_ID % 20))

topology=${topologies[$topo_index]}

echo "Running topology=$topology seed=$seed"

python "NetPPO-CausalProbe_v10.13.py" \
  --topology $topology \
  --seed $seed
