#!/bin/bash
#SBATCH --job-name=AR_space
#SBATCH --mail-type=ALL   
#SBATCH --mail-user=butlerj@berkeley.edu
#SBATCH -p high
#SBATCH -o space_$1.out #File to which standard out will be written
#SBATCH -e space_$1.err #File to which standard err will be written

source ~/.bashrc
conda activate antarctic_ars_dataset

cd ..

python clustering.py --save_path '/scratch/users/butlerj/extreme_antarctic_ars/catalog_runs' --synoptic_scale_frac $1
