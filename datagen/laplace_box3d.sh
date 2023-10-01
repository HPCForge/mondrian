#!/bin/bash
#SBATCH -p free
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0:30:00

N=10

python mesh/box3d/generator.py \
	-n $N \
	--outdir /pub/afeeney/neural-schwarz-data/box3d-mesh

build/laplace_homog_bc.out \
	--mesh /pub/afeeney/neural-schwarz-data/box3d-mesh \
	--outdir /pub/afeeney/neural-schwarz-data/box3d-mesh-sol \
