from pathlib import Path
import os
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "run_refined_joint_interaction.slurm"
FINAL_SCRIPT = REPO_ROOT / "scripts" / "finalize_refined_joint_interaction.slurm"


def test_joint_array_requests_three_hundred_single_core_tasks_without_partition_guess():
    text = ARRAY_SCRIPT.read_text()
    assert "#SBATCH --array=0-299%16" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=4G" in text
    assert "--partition" not in text
    assert "--account" not in text


def test_joint_array_maps_tasks_to_fifty_cells_and_six_replication_blocks():
    text = ARRAY_SCRIPT.read_text()
    assert "BLOCK_SIZE=50" in text
    assert "BLOCKS_PER_CELL=6" in text
    assert "CELL_INDEX=$((SLURM_ARRAY_TASK_ID / BLOCKS_PER_CELL))" in text
    assert "BLOCK_INDEX=$((SLURM_ARRAY_TASK_ID % BLOCKS_PER_CELL))" in text
    assert "START=$((BLOCK_INDEX * BLOCK_SIZE))" in text
    assert "STOP=$((START + BLOCK_SIZE))" in text
    assert '--cell-index "$CELL_INDEX"' in text


def test_joint_array_reproduces_environment_limits_threads_and_keeps_guard():
    text = ARRAY_SCRIPT.read_text()
    assert "module load python/3.12.6" in text
    assert "source .venv/bin/activate" in text
    assert "unset PYTHONPATH" in text
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in text
    assert "--allow-login-node" not in text
    assert "--no-resume" not in text


def test_joint_finalizer_is_separate_single_task_compute_job():
    text = FINAL_SCRIPT.read_text()
    assert "#SBATCH --array" not in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "finalize_refined_joint_interaction.py" in text
    assert "--allow-login-node" not in text


def test_joint_cli_help_runs_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    production = subprocess.run(
        [sys.executable, "scripts/run_refined_joint_interaction.py", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    finalizer = subprocess.run(
        [sys.executable, "scripts/finalize_refined_joint_interaction.py", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert production.returncode == 0, production.stderr
    assert "--cell-index" in production.stdout
    assert "--start" in production.stdout
    assert "--stop" in production.stdout
    assert finalizer.returncode == 0, finalizer.stderr
    assert "--outdir" in finalizer.stdout
