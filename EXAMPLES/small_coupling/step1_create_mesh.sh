#!/bin/bash 
set -e 

change_par() 
{
  # get input args
  local param=$1
  local value=$2
  local file=$3

  # locate parameter
  oldstr=`grep "^$param " $file`
  newstr="$param           =     $value"

  sed  "s?$oldstr?$newstr?g" $file  > $file.temporary
  mv $file.temporary $file
}


source  config.env

NPROC=192

rm -rf OUTPUT_FILES DATABASES_MPI MESH
mkdir -p OUTPUT_FILES DATABASES_MPI MESH
cp DATA/Par_file.init DATA/Par_file

# create mesh 
$SEM_PATH/bin/xmeshfem3D

# decompose mesh
$SEM_PATH/bin/xdecompose_mesh $NPROC MESH/ DATABASES_MPI/

# create databases
# change "NPROC  =  " in Par_file to "NPROC  =  $NPROC"
change_par NPROC "$NPROC" "DATA/Par_file"