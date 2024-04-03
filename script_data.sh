#!/bin/bash

#SBATCH -p free
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --mem=128G
#SBATCH -J datagen
#SBATCH -o datagen.out

source ~/.bashrc
conda activate nsch
python datagen/fdm/diffusion.py