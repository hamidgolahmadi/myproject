import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_summary_655210.csv")

plt.figure(figsize=(7,5))
df.boxplot(column="r_probe_used", by="topology")
plt.title("Probe reward by topology")
plt.suptitle("")
plt.ylabel("Probe reward")

plt.savefig("reward_boxplot.png", dpi=300, bbox_inches="tight")
print("saved reward_boxplot.png")
