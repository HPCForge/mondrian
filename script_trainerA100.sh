#!/bin/bash

#SBATCH -p free-gpu
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH -J nscha
#SBATCH -o nscha.out
#SBATCH -t 06:00:00

source ~/.bashrc
export PYTHONPATH=$(pwd)
conda activate nsch
python -u mondrian_lib/trainer/allen_cahn_trainer_deq.py
