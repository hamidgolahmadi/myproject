#!/bin/bash
#SBATCH --job-name=refined-d045-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=results/refined/confirmatory_production/finalize-%j.out
#SBATCH --error=results/refined/confirmatory_production/finalize-%j.err

set -euo pipefail

PROJECT_ROOT=/iridisfs/home/hg2e25/projects/myproject
OUTPUT_DIR="$PROJECT_ROOT/results/refined/confirmatory_production"
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
    echo "ERROR: tracked working-tree changes detected; refusing D045 finalization." >&2
    exit 2
fi

printf '%s\n' '=============================================================================='
echo "d045_finalize_job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "start_time=$(date --iso-8601=seconds)"
python --version
printf '%s\n' '=============================================================================='

python scripts/finalize_refined_confirmatory_production.py --outdir "$OUTPUT_DIR"

echo "end_time=$(date --iso-8601=seconds)"
echo "D045 finalization job completed successfully."
