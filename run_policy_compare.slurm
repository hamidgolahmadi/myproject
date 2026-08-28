#!/bin/bash
#SBATCH --job-name=policy_cmp
#SBATCH --output=logs/policy_compare_%j.out
#SBATCH --error=logs/policy_compare_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e

cd /iridisfs/home/hg2e25/projects/myproject

source /home/hg2e25/venvs/netppo/bin/activate

python 03_run_policy_compare.py --topologies_dir topologies --horizon 400
