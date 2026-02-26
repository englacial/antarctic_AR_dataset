#!/bin/bash

# running the optimal parameters first
sbatch run_optimal_clustering.sh

# running time perturbations
params=(9 15 18 21 24)

for param in "${params[@]}"; do
    echo "Submitting job with time parameter perturbation: $param"
    sbatch run_time_perturb.sh "$param"
done

# running space perturbations
params=(0.75 1 0.25)
for param in "${params[@]}"; do
    echo "Submitting job with spatial parameter perturbation: $param"
    sbatch run_space_perturb.sh "$param"
done

# running seed perturbations
params=(1111 2222 3333 4444 5555)
for param in "${params[@]}"; do
    echo "Submitting job with seed perturbation: $param"
    sbatch run_seed_perturb.sh "$param"
done

# running minpts perturbations
params=(3 8 10)
for param in "${params[@]}"; do
    echo "Submitting job with minpts perturbation: $param"
    sbatch run_minpts_perturb.sh "$param"
done

# running reppts perturbations
params=(5 15)
for param in "${params[@]}"; do
    echo "Submitting job with reppts perturbation: $param"
    sbatch run_reppts_perturb.sh "$param"
done
