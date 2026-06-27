#!/bin/bash
#SBATCH --job-name=generate_databases
#SBATCH --output=generate_databases.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:12:00
#SBATCH --partition=debug
#SBATCH --account=rrg-liuqy

set -e 

source config.env

# get NPROC from DATA/Par_file
NPROC=`grep "^NPROC " DATA/Par_file | awk '{print $3}'`

echo ""
echo "Generating databases with $NPROC MPI processes..."
echo ""
mpirun -np $NPROC $SEM_PATH/bin/xgenerate_databases

# interpolate model with ak135 
echo ""
echo "Interpolating model with ak135..."
echo ""
python interpolate_model.py

# regenerate_database
echo ""
echo "Re-generating databases with $NPROC MPI processes..."
echo ""
change_par MODEL gll DATA/Par_file
mpirun -np $NPROC $SEM_PATH/bin/xgenerate_databases

#create free surface nodes file for post-processing
echo ""
echo "Generating free surface nodes file for post-processing with $NPROC MPI processes..."
echo ""
mpirun -np $NPROC python create_free_surface_fields.py

# create stations
echo ""
echo "Generating stations files"
echo ""
python generate_2d_stations.py

