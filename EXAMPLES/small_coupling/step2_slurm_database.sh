#!/bin/bash
#SBATCH --job-name=generate_databases
#SBATCH --output=generate_databases.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:05:00
#SBATCH --partition=debug
#SBATCH --account=rrg-liuqy

set -e 

source config.env

# get NPROC from DATA/Par_file
NPROC=`grep "^NPROC " DATA/Par_file | awk '{print $3}'`

mpirun -np $NPROC $SEM_PATH/bin/xgenerate_databases

# create free surface nodes file for post-processing
mpirun -np $NPROC python create_free_surface_fields.py

