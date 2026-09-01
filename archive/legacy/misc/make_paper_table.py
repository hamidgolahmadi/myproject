import pandas as pd

df = pd.read_csv("results_summary_655210.csv")

table = df.groupby("topology").agg(
    runs=("seed","count"),
    reward_mean=("r_probe_used","mean"),
    reward_std=("r_probe_used","std"),
    gini_mean=("gini","mean"),
    gini_std=("gini","std"),
    dP_mean=("dP","mean"),
    risk_mean=("riskS_clip","mean"),
)

print(table)

table.to_csv("paper_table_results.csv")
