#!/bin/bash
#SBATCH --job-name=baseline_np
#SBATCH --output=logs/baseline_no_policy_%j.out
#SBATCH --error=logs/baseline_no_policy_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e

cd /iridisfs/home/hg2e25/projects/myproject

source /home/hg2e25/venvs/netppo/bin/activate

python 02_run_baseline_no_policy.py --topologies_dir topologies --horizon 400
