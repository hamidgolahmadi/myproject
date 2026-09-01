import re
import glob
import os
import pandas as pd

# ==========
# SETTINGS
# ==========
JOB_ID = "655210"
LOG_PATTERN = f"logs/netppo14_{JOB_ID}_*.out"
OUTPUT_CSV = f"results_summary_{JOB_ID}.csv"

# ==========
# REGEX
# ==========
iter_re = re.compile(r"iter=(\d+)")
probe_re = re.compile(r"r_probe_used=([-+]?\d*\.?\d+)")
gini_re = re.compile(r"gini=([-+]?\d*\.?\d+)")
dp_re = re.compile(r"dP=([-+]?\d*\.?\d+)")
risk_re = re.compile(r"riskS_clip=([-+]?\d*\.?\d+)")
topo_seed_re = re.compile(r"Running topology=(\w+) seed=(\d+)")

rows = []

files = sorted(glob.glob(LOG_PATTERN))
if not files:
    raise FileNotFoundError(f"No log files found for pattern: {LOG_PATTERN}")

for fpath in files:
    fname = os.path.basename(fpath)

    # array id from filename: netppo14_JOBID_ARRAYID.out
    m_arr = re.search(rf"netppo14_{JOB_ID}_(\d+)\.out", fname)
    if not m_arr:
        continue
    array_id = int(m_arr.group(1))

    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    topology = None
    seed = None

    # 1) try to read topology+seed from echo line
    for line in lines:
        m = topo_seed_re.search(line)
        if m:
            topology = m.group(1)
            seed = int(m.group(2))
            break

    # 2) fallback: infer from array_id
    if topology is None or seed is None:
        if array_id < 20:
            topology = "random_fixed"
            seed = array_id
        elif array_id < 40:
            topology = "scale_free"
            seed = array_id - 20
        else:
            topology = "small_world"
            seed = array_id - 40

    # find all lines with iter=
    iter_lines = [line.strip() for line in lines if "iter=" in line]

    if not iter_lines:
        rows.append({
            "job_id": JOB_ID,
            "array_id": array_id,
            "topology": topology,
            "seed": seed,
            "last_iter": None,
            "r_probe_used": None,
            "gini": None,
            "dP": None,
            "riskS_clip": None,
            "status": "no_iter_lines",
            "file": fname
        })
        continue

    last_line = iter_lines[-1]

    def extract(pattern, text):
        m = pattern.search(text)
        return float(m.group(1)) if m else None

    last_iter_m = iter_re.search(last_line)
    last_iter = int(last_iter_m.group(1)) if last_iter_m else None

    row = {
        "job_id": JOB_ID,
        "array_id": array_id,
        "topology": topology,
        "seed": seed,
        "last_iter": last_iter,
        "r_probe_used": extract(probe_re, last_line),
        "gini": extract(gini_re, last_line),
        "dP": extract(dp_re, last_line),
        "riskS_clip": extract(risk_re, last_line),
        "status": "ok",
        "file": fname
    }
    rows.append(row)

df = pd.DataFrame(rows).sort_values(["topology", "seed"]).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved summary to: {OUTPUT_CSV}")
print()
print(df.head(10).to_string(index=False))
print()
print("Counts by topology:")
print(df.groupby("topology").size())
print()
print("Mean metrics by topology:")
print(df.groupby("topology")[["r_probe_used", "gini", "dP", "riskS_clip"]].mean())
