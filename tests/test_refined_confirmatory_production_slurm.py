from pathlib import Path
import os
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "run_refined_confirmatory_production.slurm"
FINAL_SCRIPT = REPO_ROOT / "scripts" / "finalize_refined_confirmatory_production.slurm"


def test_production_array_requests_ten_single_core_tasks_without_partition_guess():
    text = ARRAY_SCRIPT.read_text()
    assert "#SBATCH --array=0-9" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=4G" in text
    assert "--partition" not in text
    assert "--account" not in text


def test_array_maps_each_task_to_exactly_one_hundred_replications():
    text = ARRAY_SCRIPT.read_text()
    assert "TASK_SIZE=100" in text
    assert "START=$((SLURM_ARRAY_TASK_ID * TASK_SIZE))" in text
    assert "STOP=$((START + TASK_SIZE))" in text
    assert '--start "$START"' in text
    assert '--stop "$STOP"' in text


def test_array_reproduces_python_environment_and_limits_threads():
    text = ARRAY_SCRIPT.read_text()
    assert "module load python/3.12.6" in text
    assert "source .venv/bin/activate" in text
    assert "unset PYTHONPATH" in text
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in text


def test_array_does_not_bypass_login_node_guard_or_disable_resume():
    text = ARRAY_SCRIPT.read_text()
    assert "--allow-login-node" not in text
    assert "--no-resume" not in text


def test_finalizer_is_a_separate_single_task_job():
    text = FINAL_SCRIPT.read_text()
    assert "#SBATCH --array" not in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "finalize_refined_confirmatory_production.py" in text
    assert "--allow-login-node" not in text


def test_production_cli_help_runs_without_pythonpath():
    completed = subprocess.run(
        [sys.executable, "scripts/run_refined_confirmatory_production.py", "--help"],
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--start" in completed.stdout
    assert "--stop" in completed.stdout


def test_finalizer_cli_help_runs_without_pythonpath():
    completed = subprocess.run(
        [sys.executable, "scripts/finalize_refined_confirmatory_production.py", "--help"],
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--outdir" in completed.stdout
