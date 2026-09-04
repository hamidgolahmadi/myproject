#!/bin/bash
#SBATCH --job-name=refined-d045
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=results/refined/confirmatory_production/slurm-%A_%a.out
#SBATCH --error=results/refined/confirmatory_production/slurm-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/iridisfs/home/hg2e25/projects/myproject
OUTPUT_DIR="$PROJECT_ROOT/results/refined/confirmatory_production"
TASK_SIZE=100
START=$((SLURM_ARRAY_TASK_ID * TASK_SIZE))
STOP=$((START + TASK_SIZE))

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
    echo "ERROR: tracked working-tree changes detected; refusing D045 production." >&2
    exit 2
fi

printf '%s\n' '=============================================================================='
echo "d045_array_job_id=${SLURM_ARRAY_JOB_ID:-unknown}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-unknown}"
echo "host=$(hostname)"
echo "git_commit=$(git rev-parse HEAD)"
echo "start_time=$(date --iso-8601=seconds)"
echo "replication_range=[$START,$STOP)"
python --version
printf '%s\n' '=============================================================================='

python scripts/run_refined_confirmatory_production.py \
    --start "$START" \
    --stop "$STOP" \
    --outdir "$OUTPUT_DIR" \
    --progress-every 10

echo "end_time=$(date --iso-8601=seconds)"
echo "D045 array task completed successfully."
