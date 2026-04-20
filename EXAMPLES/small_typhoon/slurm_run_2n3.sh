#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=run.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:12:00
#SBATCH --partition=debug
#SBATCH --account=rrg-liuqy

set -e

bash step2_slurm_database.sh

bash step3_slurm_solver.sh