from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_SCRIPT = REPO_ROOT / "scripts" / "run_refined_market_calibration.slurm"


def _text() -> str:
    return SLURM_SCRIPT.read_text(encoding="utf-8")


def test_slurm_script_requests_single_process_resources():
    text = _text()
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=8G" in text


def test_slurm_script_does_not_guess_partition_or_account():
    text = _text()
    assert "#SBATCH --partition=" not in text
    assert "#SBATCH --account=" not in text


def test_slurm_script_recreates_iridis_python_environment():
    text = _text()
    assert "module load python/3.12.6" in text
    assert "source .venv/bin/activate" in text
    assert "unset PYTHONPATH" in text
    assert "OPENBLAS_NUM_THREADS=1" in text


def test_slurm_script_runs_resumable_final_calibration_without_login_override():
    text = _text()
    assert "python scripts/run_refined_market_calibration.py" in text
    assert "--stage all" in text
    assert "--allow-login-node" not in text
    assert "--no-resume" not in text
