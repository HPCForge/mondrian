#!/bin/bash -l

#SBATCH --job-name=test      ## Name of the job.
#SBATCH -p standard          ## partition/queue name
#SBATCH --nodes=1            ## (-N) number of nodes to use
#SBATCH --ntasks=1           ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=1    ## number of cores the job needs
#SBATCH --mem=500            ## 500Mb of memory
#SBATCH --time=00:02:00      ## run time 2 min
#SBATCH --error=slurm-%J.err ## error log file

echo "hi"
