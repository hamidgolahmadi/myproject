# run_oat_postprocessing.py
# -------------------------------------------------------------
# Run all OAT post-processing steps in order
# -------------------------------------------------------------

import subprocess
import sys


SCRIPTS = [
    "merge_oat_chunks.py",
    "summarize_oat_results.py",
    "compare_topologies_oat.py",
    "detect_network_importance_oat.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        result = subprocess.run([sys.executable, script], check=True)
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed")

    print("\nAll OAT post-processing steps completed successfully.")


if __name__ == "__main__":
    main()
