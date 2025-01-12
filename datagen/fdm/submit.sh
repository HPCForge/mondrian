#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00

source ~/.bashrc
micromamba activate mondrian-fdm-data

python allen_cahn.py \
        --sim_res 128 \
        --down_res 32 \
        --num_sims 20000