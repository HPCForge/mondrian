#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:A30:1
#SBATCH --time=02:00:00

module load anaconda/2022.05
source ~/env/neural-schwarz.sh

python src/fdm/trainer.py
