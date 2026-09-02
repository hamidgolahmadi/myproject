#!/bin/bash
#SBATCH --job-name=refined_d042_cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/iridisfs/home/hg2e25/projects/myproject/results/refined/market_calibration/slurm-%j.out
#SBATCH --error=/iridisfs/home/hg2e25/projects/myproject/results/refined/market_calibration/slurm-%j.err

set -euo pipefail

PROJECT_ROOT=/iridisfs/home/hg2e25/projects/myproject
OUTPUT_DIR="$PROJECT_ROOT/results/refined/market_calibration"

cd "$PROJECT_ROOT"

module load python/3.12.6
source .venv/bin/activate
unset PYTHONPATH

# The production calibration is single-process. Prevent numerical libraries
# from silently creating extra threads beyond the requested Slurm allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"

BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "refined-model" ]]; then
    echo "ERROR: expected branch refined-model, found $BRANCH" >&2
    exit 2
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked working-tree changes detected; refusing production calibration." >&2
    exit 2
fi

echo "=== Refined D042 production calibration ==="
echo "start_time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
echo "git_commit=$(git rev-parse HEAD)"
echo "python=$(python --version 2>&1)"
echo "output_dir=$OUTPUT_DIR"
echo

python scripts/run_refined_market_calibration.py \
    --stage all \
    --outdir "$OUTPUT_DIR" \
    --progress-every 10

echo
echo "end_time=$(date --iso-8601=seconds)"
echo "Calibration job completed successfully."
