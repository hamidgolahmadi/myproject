#!/bin/bash
#SBATCH --job-name=causalprobe
#SBATCH --output=causalprobe.out
#SBATCH --error=causalprobe.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject

python "NetPPO-CausalProbe_v10.13.py"
