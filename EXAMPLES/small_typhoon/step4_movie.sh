#!/bin/bash

set -e

source config.env
NRPOC=`grep "^NPROC " DATA/Par_file | awk '{print $3}'`
NPROC_MINUS_ONE=$(($NRPOC - 1))

for f in DATABASES_MPI/proc000000_velocity_Z_*.bin  ;
#for f in DATABASES_MPI/proc000000_velocity_Z_it004000.bin;
do
  filename=$(basename $f)
  vtkname=`echo $filename | cut -d'_' -f2,3,4 |cut -d'.' -f1`
  #echo $vtkname
  $SEM_PATH/bin/xcombine_vol_data_vtk 0 $NPROC_MINUS_ONE $vtkname ./DATABASES_MPI/ ./  0
done 