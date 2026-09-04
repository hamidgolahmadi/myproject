from pathlib import Path
import os
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "run_refined_alpha_sweep.slurm"
FINAL_SCRIPT = REPO_ROOT / "scripts" / "finalize_refined_alpha_sweep.slurm"


def test_alpha_sweep_array_shape_and_mapping_are_frozen():
    text = ARRAY_SCRIPT.read_text()
    assert "#SBATCH --array=0-47%16" in text
    assert "BLOCK_SIZE=50" in text
    assert "BLOCKS_PER_ALPHA=6" in text
    assert "ALPHA_INDEX=$((SLURM_ARRAY_TASK_ID / BLOCKS_PER_ALPHA))" in text
    assert "BLOCK_INDEX=$((SLURM_ARRAY_TASK_ID % BLOCKS_PER_ALPHA))" in text
    assert "START=$((BLOCK_INDEX * BLOCK_SIZE))" in text
    assert "STOP=$((START + BLOCK_SIZE))" in text


def test_alpha_sweep_array_uses_single_core_resources_without_partition_guess():
    text = ARRAY_SCRIPT.read_text()
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=4G" in text
    assert "--partition" not in text
    assert "--account" not in text
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in text


def test_alpha_sweep_array_keeps_resume_and_login_guards_active():
    text = ARRAY_SCRIPT.read_text()
    assert "--allow-login-node" not in text
    assert "--no-resume" not in text
    assert '--alpha-index "$ALPHA_INDEX"' in text
    assert '--start "$START"' in text
    assert '--stop "$STOP"' in text


def test_alpha_sweep_finalizer_is_separate_single_task_job():
    text = FINAL_SCRIPT.read_text()
    assert "#SBATCH --array" not in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "finalize_refined_alpha_sweep.py" in text
    assert "--allow-login-node" not in text


def test_alpha_sweep_clis_show_help_without_pythonpath():
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    for script, expected in (
        ("scripts/run_refined_alpha_sweep.py", "--alpha-index"),
        ("scripts/finalize_refined_alpha_sweep.py", "--outdir"),
    ):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert expected in completed.stdout
