#!/bin/bash
#SBATCH -p free
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0:30:00

python generator.py \
	-n 20 \
	--outdir /pub/afeeney/neural-schwarz-data/box3d-mesh
