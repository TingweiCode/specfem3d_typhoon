#!/bin/bash
#SBATCH --job-name=generate_databases
#SBATCH --output=step3.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:12:00
#SBATCH --partition=debug
#SBATCH --account=rrg-liuqy

set -e 

source config.env

# get NPROC from DATA/Par_file
NPROC=`grep "^NPROC " DATA/Par_file | awk '{print $3}'`

mpirun -np $NPROC $SEM_PATH/bin/xspecfem3D

