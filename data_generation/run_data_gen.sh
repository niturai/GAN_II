#!/bin/bash
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH -J data_gen


module load python

python -m generate_ellipsoids --path ~/output/
