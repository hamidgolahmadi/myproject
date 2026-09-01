
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out.log
#SBATCH --error=err.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load python/3.12.6
source ~/venvs/netppo/bin/activate

cd ~/projects/myproject

python NetPPO-CausalProbe_v10.13.py

