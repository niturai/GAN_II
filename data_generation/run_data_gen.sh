#!/bin/bash
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=25:00:00
#SBATCH --mem=32GB
#SBATCH -J data_gen


module load python
conda activate new_env

python -m generate_ellipsoids --path /global/cfs/projectdirs/m2676/users/yuvand/GAN_III/
