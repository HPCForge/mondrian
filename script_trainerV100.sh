#!/bin/bash

#SBATCH -p free-gpu
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=64G
#SBATCH -J nschv
#SBATCH -o nschv.out
#SBATCH -t 06:00:00

source ~/.bashrc
export PYTHONPATH=$(pwd)
conda activate nsch
python -u src/torchdeq_fdm.py