#!/bin/bash
#SBATCH --job-name=refined-d048-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=results/refined/gamma_sweep/finalize-%j.out
#SBATCH --error=results/refined/gamma_sweep/finalize-%j.err

set -euo pipefail

PROJECT_ROOT=/iridisfs/home/hg2e25/projects/myproject
OUTPUT_DIR="$PROJECT_ROOT/results/refined/gamma_sweep"
cd "$PROJECT_ROOT"

module load python/3.12.6
source .venv/bin/activate
unset PYTHONPATH

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "refined-model" ]]; then
    echo "ERROR: expected branch refined-model, found $BRANCH" >&2
    exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked working-tree changes detected; refusing D048 finalization." >&2
    exit 2
fi

printf '%s\n' '=============================================================================='
echo "d048_finalize_job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "start_time=$(date --iso-8601=seconds)"
python --version
printf '%s\n' '=============================================================================='

python scripts/finalize_refined_gamma_sweep.py --outdir "$OUTPUT_DIR"

echo "end_time=$(date --iso-8601=seconds)"
echo "D048 finalization job completed successfully."
