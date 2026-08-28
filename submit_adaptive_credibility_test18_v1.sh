#!/bin/bash

set -euo pipefail

topologies=(
  "random_fixed_extreme"
  "scale_free_extreme"
  "small_world_clustered"
)

betas=(
  "0.0"
  "2.0"
  "5.0"
)

gammas=(
  "0.0"
  "0.5"
)

for topo in "${topologies[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      echo "Submitting: TOPOLOGY=$topo  BETA=$beta  GAMMA=$gamma"
      TOPOLOGY="$topo" BETA="$beta" GAMMA="$gamma" sbatch run_adaptive_credibility_grid_v1.slurm
    done
  done
done
