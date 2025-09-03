#!/bin/bash
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --time=00:08:00
#SBATCH --mem=64GB
#SBATCH -J data_gen
#SBATCH --account=m2676


module load python
conda activate new_env

python train.py --path /global/cfs/projectdirs/m2676/users/yuvand/GAN/