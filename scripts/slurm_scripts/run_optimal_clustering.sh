#!/bin/bash
#SBATCH --job-name=AR_optimal
#SBATCH --mail-type=ALL   
#SBATCH --mail-user=butlerj@berkeley.edu
#SBATCH -p high
#SBATCH -o optimal.out #File to which standard out will be written
#SBATCH -e optimal.err #File to which standard err will be written

source ~/.bashrc
conda activate antarctic_ars

cd ..

python clustering.py --save_path '/scratch/users/butlerj/extreme_antarctic_ars/catalog_runs'
