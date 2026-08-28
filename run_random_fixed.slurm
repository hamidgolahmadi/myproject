#!/bin/bash
#SBATCH --job-name=netppo_rand
#SBATCH --output=logs/netppo_rand_%j.out
#SBATCH --error=logs/netppo_rand_%j.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject
mkdir -p logs

python "NetPPO-CausalProbe_v10.13.py" \
  --topology random_fixed \
  --seed 0
