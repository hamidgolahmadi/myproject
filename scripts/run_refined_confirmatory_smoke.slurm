#!/bin/bash
#SBATCH --job-name=refined-confirm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=results/refined/confirmatory_smoke/slurm-%j.out
#SBATCH --error=results/refined/confirmatory_smoke/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=/iridisfs/home/hg2e25/projects/myproject
cd "$PROJECT_ROOT"
mkdir -p results/refined/confirmatory_smoke

module load python/3.12.6
source .venv/bin/activate
unset PYTHONPATH

printf '%s\n' '=============================================================================='
echo "confirmatory_smoke_job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "start_time=$(date --iso-8601=seconds)"
python --version
printf '%s\n' '=============================================================================='

python scripts/run_refined_confirmatory_smoke.py

echo "end_time=$(date --iso-8601=seconds)"
echo "Confirmatory smoke job completed successfully."
