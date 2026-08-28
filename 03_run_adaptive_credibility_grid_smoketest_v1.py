#topo_dir = "topologies_extreme"import os
import os
import numpy as np
import pandas as pd

from env_adaptive_credibility_v1 import InfoNetworkAdaptiveEnv

def load_topology(path):
    return np.load(path)["P"]

def run_single(P, beta, gamma, seed, horizon):
    env = InfoNetworkAdaptiveEnv(
        P_init=P,
        beta=beta,
        gamma=gamma,
        seed=seed,
        horizon=horizon,
    )

    rows = []
    done = False
    while not done:
        info, done = env.step()
        rows.append(info)

    df = pd.DataFrame(rows)
    return {
        "beta": beta,
        "gamma": gamma,
        "seed": seed,
        "return_vol": df["return"].std(),
        "riskS_mean": df["riskS"].mean(),
        "belief_var_mean": df["belief_var"].mean(),
    }

def main():
    topo_dir = "extreme_topologies_v2"
    outdir = "adaptive_grid_smoketest_v1"
    os.makedirs(outdir, exist_ok=True)

    topologies = {
        "random": f"{topo_dir}/random_fixed_extreme.npz",
        "scale_free": f"{topo_dir}/scale_free_extreme.npz",
        "small_world": f"{topo_dir}/small_world_clustered.npz",
    }

    betas = [0.0, 2.0, 5.0]
    gammas = [0.0, 0.5, 0.9]
    seeds = range(5)
    horizon = 300

    all_rows = []

    for topo_name, path in topologies.items():
        P = load_topology(path)
        for beta in betas:
            for gamma in gammas:
                for seed in seeds:
                    res = run_single(P, beta, gamma, seed, horizon)
                    res["topology"] = topo_name
                    all_rows.append(res)

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{outdir}/grid_results_smoketest.csv", index=False)
    print(df.head())
    print("\nRows:", len(df))

if __name__ == "__main__":
    main()
