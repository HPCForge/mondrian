#!/bin/bash
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=03:00:00

module load anaconda/2022.05
source ~/env/neural-schwarz.sh

python allen_cahn.py
